import logging
from bisect import bisect

import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_transformers.tokenization_bert import BertTokenizer
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

##
from mmt._image_features_reader import ImageFeaturesH5Reader
from mmt.losses import LossMap, ce_loss
from mmt.vqa_dataset import VQAClassificationDataset
#from mmt.mmt import BertImageEmbeddings
from mmt.model import ADVContrastiveProjection
from tools.registry import registry

logger = logging.getLogger(__name__)


def clip_gradients(model, max_grad_l2_norm, clip_norm_mode):
    if max_grad_l2_norm is not None:
        if clip_norm_mode == "all":
            norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_l2_norm)
        elif clip_norm_mode == "question":
            question_embedding = model.module.question_embedding_module
            norm = nn.utils.clip_grad_norm(
                question_embedding.parameters(), max_grad_l2_norm
            )
        else:
            raise NotImplementedError(
                "Clip norm mode %s not implemented" % clip_norm_mode
            )


def get_optim_scheduler(
    task_cfg,
    optimizer_grouped_parameters,
    base_lr,
):
    optimizer = Adam(optimizer_grouped_parameters, lr=base_lr)   ##
    warmup_iters = task_cfg["warmup_iters"]
    warmup_factor = task_cfg["warmup_factor"]
    lr_decay_iters = task_cfg["lr_decay_iters"]
    lr_decay = task_cfg["lr_decay"]

    def lr_update(_iter):
        if _iter <= warmup_iters:
            alpha = float(_iter) / float(warmup_iters)
            return warmup_factor * (1.0 - alpha) + alpha
        else:
            idx = bisect(lr_decay_iters, _iter)
            return pow(lr_decay, idx)

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_update)
    return optimizer, warmup_scheduler


def to_device(batch_dict, device):
    if device.type == "cpu":
        return
    for batch in batch_dict:
        for key, value in batch.items():
            if key in ["image_id", "question_id"]:
                continue
            if isinstance(value, torch.Tensor):
                batch[key] = value.cuda(device=device, non_blocking=True)


def forward_eval(device, batch_dict, model, revqa_eval=False, revqa_split="revqa"):
    batch_size = len(batch_dict[0]["question_id"])
    for batch in batch_dict:
        results_dict = run_model(batch, model, device)
        batch.update(results_dict)

    loss, batch_score = ce_loss(
        batch_dict[0], device, val_run=True, revqa_eval=revqa_eval, split=revqa_split
    )

    # evaluation logging
    if registry.get("eval_only", False):
        return batch_dict[0]

    del results_dict
    del batch_dict

    return float(loss), float(batch_score), batch_size


def get_batch(dataloaders, dkey):    ##dkey: train_ce, ..     
    ikey = dkey + "_iter"             ##'train_ce_iter'
    load_epoch = ikey not in dataloaders
    
    if not load_epoch:
        batch_dicts = next(dataloaders[ikey], None)
        if batch_dicts is None:
            load_epoch = True

    if load_epoch:    ##true
        dataloaders[ikey] = iter(dataloaders[dkey])
        batch_dicts = next(dataloaders[ikey], None)      
        assert batch_dicts is not None

    return batch_dicts

def generate_adv(self, dec_hiddens, lm_labels):
        dec_hiddens = dec_hiddens.detach()

        dec_hiddens.requires_grad = True

        lm_logits = self.t5_model.lm_head(dec_hiddens)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        loss = criterion(lm_logits.view(-1, lm_logits.size(-1)),
                         lm_labels.view(-1))
        loss.backward()
        dec_grad = dec_hiddens.grad.detach()
        l2_norm = torch.norm(dec_grad, dim=-1)

        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturbed_dec = dec_hiddens + self.neg_eps * dec_grad.detach()
        perturbed_dec = perturbed_dec  # [b,t,d]

        self.zero_grad()

        return perturbed_dec

def generate_cont_adv(self, enc_hiddens, enc_mask, dec_hiddens, dec_mask, lm_logits, tau, eps):
    enc_hiddens = enc_hiddens.detach()
    dec_hiddens = dec_hiddens.detach()
    lm_logits = lm_logits.detach()
    dec_hiddens.requires_grad = True

    avg_enc = self.avg_pool(self.projection(enc_hiddens), enc_mask)

    avg_dec = self.avg_pool(self.projection(dec_hiddens), dec_mask)

    cos = nn.CosineSimilarity(dim=-1)
    logits = cos(avg_enc.unsqueeze(1), avg_dec.unsqueeze(0)) / tau

    cont_crit = nn.CrossEntropyLoss()
    labels = torch.arange(avg_enc.size(0), device=enc_hiddens.device)
    loss = cont_crit(logits, labels)
    loss.backward()

    dec_grad = dec_hiddens.grad.detach()
    l2_norm = torch.norm(dec_grad, dim=-1)
    dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

    perturb_dec_hidden = dec_hiddens + eps * dec_grad
    perturb_dec_hidden = perturb_dec_hidden.detach()
    perturb_dec_hidden.requires_grad = True
    perturb_logits = self.t5_model.lm_head(perturb_dec_hidden)

    true_probs = F.softmax(lm_logits, -1)
    true_probs = true_probs * dec_mask.unsqueeze(-1).float()

    perturb_log_probs = F.log_softmax(perturb_logits, -1)

    kl_crit = nn.KLDivLoss(reduction="sum")
    vocab_size = lm_logits.size(-1)

    kl = kl_crit(perturb_log_probs.view(-1, vocab_size), true_probs.view(-1, vocab_size))
    kl = kl / torch.sum(dec_mask).float() 
    kl.backward()
    kl_grad = perturb_dec_hidden.grad.detach()

    l2_norm = torch.norm(kl_grad, dim=-1)
    kl_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

    perturb_dec_hidden = perturb_dec_hidden - eps * kl_grad

    return perturb_dec_hidden

def run_model1(batch, model, device):
    # send to gpu
    input_keys = list(batch.keys())      
    for key in input_keys:
        if key in ["image_id", "question_id"]:        
            continue
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].cuda(device=device, non_blocking=True)
    
    results_dict = model(batch)    ##gt_answer_scores = model(batch, compute_loss=False)
    #model = ADVContrastiveProjection
    mmt_config = BertConfig.from_dict(task_cfg["MMT"])
    proj_enc_h, proj_dec_h = ADVContrastiveProjection(mmt_config)
    cos = nn.CosineSimilarity(dim=-1)
    cont_crit = nn.CrossEntropyLoss()
    sim_matrix = cos(proj_enc_h.unsqueeze(1), proj_dec_h.unsqueeze(0))
    #perturbed_dec = self.generate_adv(sequence_output, lm_labels)  # [n,b,t,d] or [b,t,d]
    perturbed_dec = generate_adv(results_dict["vil_prediction"], batch['target'])  
    batch_size = input_ids.size(0)

    proj_pert_dec_h = projection(perturbed_dec)
    #avg_pert = self.avg_pool(proj_pert_dec_h, decoder_attention_mask)

    adv_sim = cos(proj_enc_h, proj_pert_dec_h).unsqueeze(1)  # [b,1]   #sim input and out

    pos_dec_hidden = generate_cont_adv(results_dict["pooled_output"], results_dict["vil_prediction"], lm_logits, self.tau, self.pos_eps)

    avg_pos_dec = self.avg_pool(self.projection(pos_dec_hidden), decoder_attention_mask)

    pos_sim = cos(proj_enc_h, pos_dec_hidden).unsqueeze(-1)  # [b,1] #sim input and out
    logits = torch.cat([sim_matrix, adv_sim], 1) / self.tau

    identity = torch.eye(batch_size, device=input_ids.device)
    pos_sim = identity * pos_sim
    neg_sim = sim_matrix.masked_fill(identity == 1, 0)
    new_sim_matrix = pos_sim + neg_sim
    new_logits = torch.cat([new_sim_matrix, adv_sim], 1)   ##pos+neg, 

    labels = torch.arange(batch_size,  device=input_ids.device)          ##labels = batch_dict[0]["target"].argmax(dim=-1).view(-1, 1
    cont_loss = cont_crit(logits, labels)
    new_cont_loss = cont_crit(new_logits, labels)

    cont_loss = 0.5 * (cont_loss + new_cont_loss)

    # delete batch-inputs (only keep results)
    for key in input_keys:
        if key in ["image_id", "question_id", "target"]:
            continue
        else:
            del batch[key]
    return results_dict

def run_model(batch, model, device):
    # send to gpu
    input_keys = list(batch.keys())      ##['input_imgs', 'image_mask', 'image_loc', 'question_indices', 'question_mask', 'image_id', 'question_id', 'target', 'mask']
    for key in input_keys:
        if key in ["image_id", "question_id"]:        
            continue
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].cuda(device=device, non_blocking=True)
    results_dict = model(batch)    
    # delete batch-inputs (only keep results)
    for key in input_keys:
        if key in ["image_id", "question_id", "target"]:
            continue
        else:
            del batch[key]
    return results_dict

def forward_train(device, dataloaders, model, train_type):
    if train_type == "ce":
        batch_dicts = get_batch(dataloaders, "train_ce")
        # throw away rephrasings batch
        batch_dicts = batch_dicts[:1]
    elif train_type == "adv":
        batch_dicts = get_batch(dataloaders, "train_adv") 
    else:
        batch_dicts = get_batch(dataloaders, "train_scl")
    
    if train_type == "scl":
        ##for batch in batch_dicts:
        ##    results_dict = run_model(batch, model, device)
        ##    batch.update(results_dict)
        ##loss, batch_score = LossMap["SCLLoss"](batch_dicts)
        loss, batch_score = pgd_attack_opt(batch_dicts, model, device) 

    elif train_type == "ce":
        for batch in batch_dicts:
            results_dict = run_model(batch, model, device)
            batch.update(results_dict)
        loss, batch_score = ce_loss(batch_dicts, device)
    else:
        loss, batch_score = pgd_attack_opt(batch_dicts, model, device)   #pgd_attack

    del batch_dicts
    return loss, float(batch_score)
    
def load_dataset(task_cfg):     ##def build_dataloader(dataset, collate_fn, is_train, opts):
    from mmtadvorg.samplers import ContrastiveSampler, RandomSampler, ContrastiveSamplerP
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    #tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", do_lower_case=True)
    #tokenizer = BasicTokenizer.from_pretrained("pretrained-vqa2")
    #tokenizer = BasicTokenizer.from_pretrained("pretrained-vqa2", dir_data='/data1/MitraTj/Proj/Contrastive/concatVQA-adv/log/mutan_att_train/tokenizers')

    # image-features
    trainval_features = ImageFeaturesH5Reader(task_cfg["trainval_features_path"])
    test_features = ImageFeaturesH5Reader(task_cfg["test_features_path"])
    dataloaders = {}

    # one train split and multiple evaluation splits
    load_splits = [task_cfg["train_split"]] + task_cfg["val_split"]   ##train_aug +  [minval, revqa, revqa_bt, val, test]

    logger.info(f"Splits to load: {load_splits}")

    for split in load_splits:
        dataset = VQAClassificationDataset(       ##return_list
            split=split,
            image_features_reader=trainval_features
            if "test" not in split
            else test_features,
            tokenizer=tokenizer,
            extra_args=task_cfg,
        )
        #import pdb; pdb. set_trace()
        if "train" in split:
            if registry.alt_train:
                samplers = ["scl", "ce", "adv"]
            else:
                samplers = ["ce"]
        else:
            samplers = ["none"]

        # build loaders for each sampler type
        for _sampler in samplers:
            sampler_tag = f"_{_sampler}" if _sampler != "none" else ""
            if _sampler == "ce" and registry.alt_train:
                batch_size =  210   #task_cfg["batch_size"] * 2
            elif _sampler == "scl" and registry.alt_train:
                batch_size = task_cfg["batch_size"]
            elif _sampler == "adv" and registry.alt_train:
                batch_size = 40      ##40
                #batch_size = task_cfg["batch_size"]
            else: 
                batch_size = task_cfg["batch_size"]
            
            # build the sampler
            
            if _sampler == "ce":
                sampler = RandomSampler(dataset)
            elif _sampler == "scl":
                sampler = ContrastiveSampler(dataset, task_cfg, split=split)
            else:
                #sampler = ContrastiveSamplerP(dataset, task_cfg, split=split)
                #sampler = TokenBucketSampler(dataset, task_cfg, droplast=False)
                #sampler = TokenBucketSamplerForItm(dataset, task_cfg, split=split)
                sampler = RandomSampler(dataset)
                #sampler = ContrastiveSampler(dataset, task_cfg, split=split)
            '''''
            if _sampler == "scl":
                sampler = ContrastiveSampler(dataset, task_cfg, split=split)
            else:
                sampler = RandomSampler(dataset)
            '''''
            split_tag = "train" if "train" in split else split
            dataloaders[f"{split_tag}" + sampler_tag] = DataLoader(
                dataset,
                sampler=sampler,
                batch_size=batch_size,
                num_workers=registry.workers,
                pin_memory=True,
                drop_last=True if split_tag == "train" else False,
                #shuffle=False,
            )

    return dataloaders
################################################################################
        # ADV
#################################################################################   
criterion_kl = torch.nn.KLDivLoss(size_average=False)

def pgd_attack(model, device, dataloaders, logadv=False):
    #model.eval()
    batch_dicts = get_batch(dataloaders, "train_adv") 
    for batch in batch_dicts:    
        # calculate the prob. scores for clean samples
        answer_scores_gt = model(batch)
        #gt_answer_prob = F.softmax(answer_scores_gt["vil_prediction"], dim=1)
        #################################################################################
        # Image
        #################################################################################
        batch['input_imgs'] = batch['input_imgs'].cuda(device=device, non_blocking=True)
        batch['image_loc'] = batch['image_loc'].cuda(device=device, non_blocking=True)
        img_embeds_init = model.module.mmt.image_embeddings(batch['input_imgs'], batch['image_loc'],adv_training=False,adv_modality=None,adv_delta=None)  ##[40, 101, 768]
        ##img_delta = torch.rand_like(img_embeds_init) * (8. /40.) * 2 - (8. /40.)    ##[40, 101, 768]
        img_delta = img_embeds_init.detach() + 0.001 * torch.randn(img_embeds_init.shape).cuda().detach()
        #################################################################################
        # Text
        #################################################################################
        batch["question_indices"] = batch["question_indices"].cuda(device=device, non_blocking=True)
        txt_embeds_init = model.module.text_bert.embeddings.word_embeddings(batch["question_indices"])
        txt_delta = txt_embeds_init.detach() + 0.001 * torch.randn(txt_embeds_init.shape).cuda().detach()
        #import pdb; pdb. set_trace()
        for astep in range(5):     ##attack_params['num_steps']
            img_delta.requires_grad_()
            txt_delta.requires_grad_()
                    
            answer_scores1 = model(batch,adv_training=True,adv_modality=["image"],adv_delta_txt=None, adv_delta_img=img_delta)
            answer_scores2 = model(batch,adv_training=True,adv_modality=["text"],adv_delta_txt = txt_delta, adv_delta_img=None)
            #answer_logprob = F.log_softmax(batch["vil_prediction"], dim=1)
            with torch.enable_grad():       
                loss_kl1 = criterion_kl((F.log_softmax(answer_scores1["vil_prediction"], dim=1)),(F.softmax(answer_scores_gt["vil_prediction"], dim=1))) 
                loss_kl2 = criterion_kl((F.log_softmax(answer_scores2["vil_prediction"], dim=1)),(F.softmax(answer_scores_gt["vil_prediction"], dim=1)))
        
            #grad = torch.autograd.grad(loss_kl, [img_delta])[0] 
            img_delta = img_delta.clone().detach() + 1* torch.sign(loss_kl1.detach())
            img_delta = torch.min(torch.max(img_delta, img_embeds_init), img_embeds_init)
            img_delta = torch.clamp(img_delta, 0.0, 1.0)
            txt_delta = txt_delta.clone().detach() + 1* torch.sign(loss_kl2.detach())
            txt_delta = torch.min(torch.max(txt_delta, txt_embeds_init), txt_embeds_init)
            txt_delta = torch.clamp(txt_delta, 0.0, 1.0)  
    #loss_robust = (1.0/100) * criterion_kl(F.log_softmax(model(batch,adv_training=True,adv_modality=["image"],adv_delta_img=img_delta), dim=1),
    #                                               F.softmax(model(batch), dim=1))
    answer_scores_1 = model(batch,adv_training=True,adv_modality=["image"],adv_delta_txt=None, adv_delta_img=img_delta)
    loss_robust_1 = (1.0/100) * criterion_kl((F.log_softmax(answer_scores_1["vil_prediction"], dim=1)),(F.softmax(answer_scores_gt["vil_prediction"], dim=1))) 
    bce_loss_1 = F.cross_entropy(answer_scores_1["vil_prediction"], batch['target'].cuda(device=device, non_blocking=True))

    answer_scores_2 = model(batch,adv_training=True,adv_modality=["text"],adv_delta_txt = txt_delta, adv_delta_img=None)
    loss_robust_2 = (1.0/100) * criterion_kl((F.log_softmax(answer_scores_2["vil_prediction"], dim=1)),(F.softmax(answer_scores_gt["vil_prediction"], dim=1))) 
    bce_loss_2 = F.cross_entropy(answer_scores_2["vil_prediction"], batch['target'].cuda(device=device, non_blocking=True)) 
    
    loss = (bce_loss_1 + bce_loss_2 + 1.0 * (loss_robust_1+loss_robust_2)) / (5*2)

    input_keys = list(batch.keys())
    for key in input_keys:
        if key in ["image_id", "question_id"]:        
            continue
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].cuda(device=device, non_blocking=True)
    for key in input_keys:
        if key in ["image_id", "question_id", "target"]:
            continue
        else:
            del batch[key]
    batch.update(answer_scores_1)

    _, batch_score = ce_loss(batch, device) 
    
    del batch_dicts
    return loss, float(batch_score)
###########################################
def pgd_attack_opt(batch_dicts, model, device):
    eps = 16.0 / 255.0
    alpha = 1.0 / 255.0
    weight = 10.0
    steps = 30   #30
    eot_iter = 1
    #loss = nn.CrossEntropyLoss()
    #model.eval()
    #batch_dicts = get_batch(dataloaders, "train_adv") 
    for batch in batch_dicts:    
        answer_scores_gt = model(batch)
        batch['input_imgs'] = batch['input_imgs'].cuda(device=device, non_blocking=True)
        batch['image_loc'] = batch['image_loc'].cuda(device=device, non_blocking=True)
        img_embeds_init = model.module.mmt.image_embeddings(batch['input_imgs'], batch['image_loc'],adv_training=False,adv_modality=None,adv_delta=None) 
        org_embed_img = img_embeds_init.detach()
        img_delta = img_embeds_init.clone().detach() 
        #if random_start:
        img_delta = img_delta + torch.empty_like(img_delta).uniform_(-eps, eps)
        img_delta = torch.clamp(img_delta, min=0, max=1)

        batch["question_indices"] = batch["question_indices"].cuda(device=device, non_blocking=True)
        txt_embeds_init = model.module.text_bert.embeddings.word_embeddings(batch["question_indices"])
        org_embed_txt = txt_embeds_init.detach()
        txt_delta = txt_embeds_init.clone().detach() 
        txt_delta = txt_delta + torch.empty_like(txt_delta).uniform_(-eps, eps)
        txt_delta = torch.clamp(txt_delta, min=0, max=1)
        
        for _ in range(steps):
            img_delta = img_delta.detach()
            img_delta.requires_grad = True
            txt_delta = txt_delta.detach()
            txt_delta.requires_grad = True

            # apply EOT to the attacker
            eot_grads = []
            eot_grads2 = []
            # EOT is applied when eot_iter > 1
            for _ in range(eot_iter):
                if img_delta.grad:
                    img_delta.grad.zero_()
                if txt_delta.grad:
                    txt_delta.grad.zero_()
                #all_ces = []
                #all_ces2 = []
                all_regs = []
                all_regs2 = []
                answer_scores1 = model(batch,adv_training=True,adv_modality=["image"],adv_delta_txt=None, adv_delta_img=img_delta)
                answer_scores2 = model(batch,adv_training=True,adv_modality=["text"],adv_delta_txt = txt_delta, adv_delta_img=None)
                
                for i in range(len(answer_scores1["vil_prediction"])):
                    #loss_kl1 = criterion_kl((F.log_softmax(answer_scores1["vil_prediction"][i])),(F.softmax(answer_scores_gt["vil_prediction"], dim=1))) 
                    loss_kl1 = criterion_kl((torch.nn.functional.log_softmax(answer_scores1["vil_prediction"][i])),(torch.nn.functional.log_softmax(answer_scores_gt["vil_prediction"], dim=1))) 
                    all_regs.append(loss_kl1)
                for i in range(len(answer_scores2["vil_prediction"])):
                    loss_kl2 = criterion_kl((F.log_softmax(answer_scores2["vil_prediction"][i])),(F.softmax(answer_scores_gt["vil_prediction"], dim=1))) 
                    #all_ces2.append(bce_loss_2)
                    all_regs2.append(loss_kl2)
                #bce_loss_1 = F.cross_entropy(answer_scores1["vil_prediction"], batch['target'].cuda(device=device, non_blocking=True))
                #bce_loss_2 = F.cross_entropy(answer_scores2["vil_prediction"], batch['target'].cuda(device=device, non_blocking=True))
                
                loss = sum(all_regs) + sum(all_regs2)
                ##cost = 1 * loss_img   #targeted:-1
                ##grad = torch.autograd.grad(cost, img_delta, create_graph=True)[0]
                
                #grad = img_delta.clone().detach().float() 
                grad = img_delta.clone().detach() + 1* torch.sign(loss.detach())
                grad2 = txt_delta.clone().detach() + 1* torch.sign(loss.detach())
                
                eot_grads.append(grad.detach().clone())
                ##cost2 = 1 * loss_txt
                ##grad2 = torch.autograd.grad(cost2, txt_delta, create_graph=True)[0]
                eot_grads2.append(grad2.detach().clone())

            #import pdb; pdb. set_trace()
            grad = sum(eot_grads) / eot_iter
            grad2 = sum(eot_grads2) / eot_iter
            
            # adv image update, image is NOT normalized
            img_delta = img_delta.detach() + alpha * grad.sign()
            delta = torch.clamp(img_delta - org_embed_img, min=-eps, max=eps)
            img_delta = torch.clamp(org_embed_img + delta, min=0, max=1)
            
            txt_delta = txt_delta.detach() + alpha * grad2.sign()
            delta2 = torch.clamp(txt_delta - org_embed_txt, min=-eps, max=eps)
            txt_delta = torch.clamp(org_embed_txt + delta2, min=0, max=1)

        input_keys = list(batch.keys())      
        for key in input_keys:
            if key in ["image_id", "question_id"]:        
                continue
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].cuda(device=device, non_blocking=True)

        answer_scores_1_1 = model(batch,adv_training=True,adv_modality=["image"],adv_delta_txt=None, adv_delta_img=img_delta)  #results_dict 
        answer_scores_2_2 = model(batch,adv_training=True,adv_modality=["text"],adv_delta_txt = txt_delta, adv_delta_img=None) 

        loss_robust_1 = (1.0/100) * criterion_kl((F.log_softmax(answer_scores_1_1["vil_prediction"], dim=1)),(F.softmax(answer_scores_gt["vil_prediction"], dim=1))) 
        bce_loss_1 = F.cross_entropy(answer_scores_1_1["vil_prediction"], batch['target'].cuda(device=device, non_blocking=True))

        loss_robust_2 = (1.0/100) * criterion_kl((F.log_softmax(answer_scores_2_2["vil_prediction"], dim=1)),(F.softmax(answer_scores_gt["vil_prediction"], dim=1))) 
        bce_loss_2 = F.cross_entropy(answer_scores_2_2["vil_prediction"], batch['target'].cuda(device=device, non_blocking=True)) 
    
        loss = (bce_loss_1 + bce_loss_2 + 1.0 * (loss_robust_1+loss_robust_2)) / (steps*2)

        batch_scores = compute_score_with_logits(answer_scores_1_1["vil_prediction"], batch['target'].cuda(device=device, non_blocking=True), device)
        batch_score = batch_scores.sum() / len(answer_scores_1_1["vil_prediction"])
        batch_scores2 = compute_score_with_logits(answer_scores_2_2["vil_prediction"], batch['target'].cuda(device=device, non_blocking=True), device)
        batch_score2 = batch_scores2.sum() / len(answer_scores_2_2["vil_prediction"])
        batch_score = (batch_score + batch_score2) / 2.0
    #del batch_dicts
    return loss, batch_score

def compute_score_with_logits(logits, labels, device):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size())

    if device.type != "cpu":
        one_hots = one_hots.cuda()

    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * labels
    return scores

SIGMA = torch.tensor([0.229, 0.224, 0.225])
MEAN = torch.tensor([0.485, 0.456, 0.406])
upper, lower = (1. - MEAN) / SIGMA, (0. - MEAN) / SIGMA
def fgsm(model, device, dataloaders):
    eps = 16.0 / 255.0
    eps = (eps / 255.) / SIGMA
    eps = eps[None, :, None, None]
    batch_dicts = get_batch(dataloaders, "train_adv") 
    for batch in batch_dicts:    
        answer_scores_gt = model(batch)
        batch['input_imgs'] = batch['input_imgs'].cuda(device=device, non_blocking=True)
        batch['image_loc'] = batch['image_loc'].cuda(device=device, non_blocking=True)
        img_embeds_init = model.module.mmt.image_embeddings(batch['input_imgs'], batch['image_loc'],adv_training=False,adv_modality=None,adv_delta=None) 
        #img_embeds_init = img_embeds_init.clone().detach().requires_grad_(True).to(img_embeds_init.device)

        batch["question_indices"] = batch["question_indices"].cuda(device=device, non_blocking=True)
        txt_embeds_init = model.module.text_bert.embeddings.word_embeddings(batch["question_indices"])
        txt_embeds_init = txt_embeds_init.clone().detach().requires_grad_(True).to(txt_embeds_init.device)
        
        #outputs = model(x_)
        loss1 = F.cross_entropy(answer_scores_gt["vil_prediction"], batch['target'].cuda(device=device, non_blocking=True))
        loss1.backward()
        with torch.no_grad():
            adv_gradient = torch.sign(img_embeds_init.grad)
            adv_gradient *= eps
            img_delta = img_embeds_init + adv_gradient
            img_delta = torch.max(torch.min(img_delta, upper), lower)
            img_delta = img_delta.detach()

            adv_gradientTxt = torch.sign(txt_embeds_init.grad)
            adv_gradientTxt *= eps
            txt_delta = txt_embeds_init + adv_gradientTxt
            txt_delta = torch.max(torch.min(txt_delta, upper), lower)
            txt_delta = txt_delta.detach()

        answer_scores_1_1 = model(batch,adv_training=True,adv_modality=["image"],adv_delta_txt=None, adv_delta_img=img_delta)  #results_dict 
        answer_scores_2_2 = model(batch,adv_training=True,adv_modality=["text"],adv_delta_txt=txt_delta, adv_delta_img=None)

        loss_robust_1 = (1.0/100) * criterion_kl((F.log_softmax(answer_scores_1_1["vil_prediction"], dim=1)),(F.softmax(answer_scores_gt["vil_prediction"], dim=1))) 
        bce_loss_1 = F.cross_entropy(answer_scores_1_1["vil_prediction"], batch['target'].cuda(device=device, non_blocking=True))
        loss_robust_2 = (1.0/100) * criterion_kl((F.log_softmax(answer_scores_2_2["vil_prediction"], dim=1)),(F.softmax(answer_scores_gt["vil_prediction"], dim=1))) 
        bce_loss_2 = F.cross_entropy(answer_scores_2_2["vil_prediction"], batch['target'].cuda(device=device, non_blocking=True))

        loss = bce_loss_1 + bce_loss_2 + 1.0 * (loss_robust_1 + loss_robust_2)

        batch_scores = compute_score_with_logits(answer_scores_1_1["vil_prediction"], batch['target'].cuda(device=device, non_blocking=True), device)
        batch_score = batch_scores.sum() / len(answer_scores_1_1["vil_prediction"])
        
    return loss, float(batch_score)
