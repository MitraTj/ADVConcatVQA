import functools
import logging
import math

import torch
import torch.nn.functional as F
from torch.autograd import Function
from pytorch_transformers.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertIntermediate,
    BertLayerNorm,
    BertOutput,
    BertPreTrainedModel,
    BertSelfOutput,
)
from torch import nn

from tools.registry import registry
from mmt.textvqa_encoders import ImageEncoder

logger = logging.getLogger(__name__)


def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return (
        0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
    )

def swish(x):
    return x * torch.sigmoid(x) 

def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):    ##def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs)
            if self.output_attentions
            else (context_layer,)
        )
        return outputs


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = (
            set(heads) - self.pruned_heads
        )  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        #nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class BertEncoder(nn.Module):     ##    TransformerEncoderLayer:  dropout -> add residual
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions         ##self.embed_dim = args.encoder_embed_dim
        self.output_hidden_states = config.output_hidden_states   
        self.layer = nn.ModuleList(                                
            [BertLayer(config) for _ in range(config.num_hidden_layers)]     ##self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        )                                                                    ##self.dropout = args.dropout

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            #import pdb; pdb. set_trace()
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class MMT(nn.Module):     ##class MMT(BertPreTrainedModel)
    """
    MMT has two transfomers MMT and TextBert.
    """
    def __init__(self, mmt_config, text_bert_config):
        #super(MMT, self).__init__( mmt_config, text_bert_config)
        super().__init__()
        # self.mmt_config = BertConfig(**self.config.mmt)
        self.mmt_config = mmt_config          ##Bertconfig
        self.text_bert_config = text_bert_config

        if self.mmt_config.finetune_ocr_obj:        ##false   changed to true
            logger.info("Finetuning object and ocr FRCN layer")
            self.frcn_encoder_type = "finetune_faster_rcnn_fpn_fc7"
        else:
            logger.info("Not finetuning object and ocr FRCN layer")
            self.frcn_encoder_type = "default"

        self.normalize = self.mmt_config.normalize     ##true
        if not self.mmt_config.normalize:
            logger.info("Not normalizing OCR and Object features")

        self.fusion_method = self.mmt_config.fusion_method
        logger.info(f"Fusion Method is : {self.fusion_method}")

        # weight-decay
        self.weight_decay = (
            mmt_config.weight_decay if hasattr(mmt_config, "weight_decay") else 0.0
        )

        self.freeze_mmt_and_textbert = (
            mmt_config.freeze_mmt_and_textbert
            if hasattr(mmt_config, "freeze_mmt_and_textbert")
            else False
        )
        if self.freeze_mmt_and_textbert:
            logger.info(f"Freezing MMT and TextBERT")

        # build the models
        self.build()

        if registry.freeze_textbert_and_mmt:
            logger.info("Freezing TextBERT and MMT")
            for name, param in self.named_parameters():
                if "linear_classifier" in name:
                    continue
                else:
                    param.requires_grad = False

    def build(self):
        # modules requiring custom learning rates (usually for finetuning)
        self.finetune_modules = []

        # split model building into several components
        self._build_txt_encoding()
        self._build_mmt()
        self._build_output()
    
        if registry.freeze_textbert_and_mmt:
            logger.info("Building a linear classifier layer")
            self.linear_classifier = LinearClassifier(3129, 3129)

    def _build_txt_encoding(self):
        TEXT_BERT_HIDDEN_SIZE = 768
        # self.text_bert_config = BertConfig(**self.config.text_bert)
        if self.text_bert_config.text_bert_init_from_bert_base:      ##true
            self.text_bert = TextBert.from_pretrained(
                "bert-base-uncased", config=self.text_bert_config
            )
            # Use a smaller learning rate on text bert when initializing
            # from BERT_BASE
            self.finetune_modules.append(
                {
                    "module": self.text_bert,
                    "lr_scale": self.text_bert_config.lr_scale_text_bert,
                }
            )
        else:
            logger.info("NOT initializing text_bert from BERT_BASE")
            self.text_bert = TextBert(self.text_bert_config)

        # if the text bert output dimension doesn't match the
        # multimodal transformer (mmt) hidden dimension,
        # add a linear projection layer between the two
        if self.mmt_config.hidden_size != TEXT_BERT_HIDDEN_SIZE:     ##768
            logger.info(
                "Projecting text_bert output to {} dim".format(
                    self.mmt_config.hidden_size
                )
            )
            self.text_bert_out_linear = nn.Linear(
                TEXT_BERT_HIDDEN_SIZE, self.mmt_config.hidden_size
            )
        else:
            self.text_bert_out_linear = nn.Identity()

    def _build_mmt(self):                  ##self.mmt = MMT(self.mmt_config)
        self.mmt = MMT_VQA(self.mmt_config)    ##BertPreTrainedModel
        
        self.finetune_modules.append(
            {
                "module": self.mmt,
                "lr_scale": self.mmt_config.lr_scale_mmt,
            }
        )
    #############
    def _build_output(self):
        # dynamic OCR-copying scores with pointer network
        #self.reversal_layer = GradReverse(-1.0)      ##lambda_grl
        self.vil_prediction = SimpleClassifier(
            self.mmt_config.hidden_size, self.mmt_config.hidden_size * 2, 3129, 0.5      ##16119
        )
        self.vil_prediction_gqa = SimpleClassifier(
            self.mmt_config.hidden_size, self.mmt_config.hidden_size * 2, 1533, 0.5     ##config.bi_hidden_size
        )

        
        self.dropout = nn.Dropout(self.mmt_config.hidden_dropout_prob)
        ##self.contrastive_projection = ContrastiveProjection(self.mmt_config)
        if hasattr(self.mmt_config, "contrastive") and self.mmt_config.contrastive in [
            "simclr",
            "better",
            "finetune",
        ]:
            self.contrastive_projection = ContrastiveProjection(self.mmt_config)
            #self.contrastive_projection = ADVContrastiveProjection(self.mmt_config)

    def _build_aux_heads(self):   
        #from mmt.vilbert import SimpleClassifier

        # spatial-category classification head
        self.origin_transform = SimpleClassifier(self.mmt_config.hidden_size, 128, 32, 0.5)
        self.dest_transform = SimpleClassifier(self.mmt_config.hidden_size, 128, 32, 0.5)
        self.spatial_classifier = nn.Linear(32, 12)

    def forward(self, batch_dict, adv_training=False, adv_modality=None, adv_delta_txt=None, adv_delta_img=None):
        self._forward_mmt_and_text(batch_dict)    ##self._forward_obj_encoding(batch_dict)
        self._forward_output(batch_dict)          ##self._forward_ocr_encoding(batch_dict)

        #if self.use_aux_heads:
        #    self._forward_aux(batch_dict)
        if registry.freeze_textbert_and_mmt:
            results_dict = {
                "vil_prediction": batch_dict["lc_out"],
                #"contastive_projection_norm": None,
                "contrastive_projection_norm": None,
                "attention_weights": None,
            }
            return results_dict

        results_dict = {
            "vil_prediction": batch_dict["vil_prediction"],
            "contrastive_projection_norm": batch_dict["contrastive_projection_norm"]
            if (
                hasattr(self.mmt_config, "contrastive")
                and self.mmt_config.contrastive in ["simclr", "better"]
            )
            else None,
            # "attention_weights": batch_dict["attention_weights"] if registry.squint_loss else None
        }
        return results_dict

    def _forward_mmt_and_text(self, batch_dict,adv_training=False, adv_modality=None, adv_delta_txt=None, adv_delta_img=None):  #_forward_mmt
        if self.freeze_mmt_and_textbert:
            self.text_bert.eval()
            self.mmt.eval()

        # first forward the text BERT layers
        text_bert_out = self.text_bert(batch_dict)   #text_bert_out = self.text_bert(batch_dict)
        batch_dict["text_bert_emb"] = self.text_bert_out_linear(text_bert_out)
        
        mmt_results = self.mmt(batch_dict, adv_training, adv_modality, adv_delta_txt, adv_delta_img)    ##sequence_output 
        batch_dict.update(mmt_results)  
        

    def _forward_output(self, batch_dict):    ##_forward_output  _forward_mmt_and_output
        if self.fusion_method == "sum":
            batch_dict["pooled_output"] = self.dropout(
                batch_dict["pooled_text_output"] + batch_dict["pooled_image_output"]  ##pooled_output = self.dropout(pooled_output_t + pooled_output_v)
            )
        elif self.fusion_method == "mul":
            batch_dict["pooled_output"] = self.dropout(
                batch_dict["pooled_text_output"] * batch_dict["pooled_image_output"]
            )
        else:
            assert False
        batch_dict["vil_prediction"] = self.vil_prediction(batch_dict["pooled_output"])    ##vil_prediction
        # batch_dict["vil_prediction_gqa"] = self.vil_prediction_gqa(batch_dict["pooled_output"])
        

        if registry.freeze_textbert_and_mmt:     ##false
            batch_dict["lc_out"] = self.linear_classifier(batch_dict["vil_prediction"])
            return

        if hasattr(self.mmt_config, "contrastive") and self.mmt_config.contrastive in [
            "simclr",
            "better",
        ]:
            self.contrastive_projection(batch_dict)
    
    def get_optimizer_parameters(self, base_lr):
        optimizer_param_groups = []
        # collect all the parameters that need different/scaled lr
        finetune_params_set = set()

        for fine_module in self.finetune_modules:
            use_wd, no_wd = [], []
            for name, param in fine_module["module"].named_parameters():
                if param.requires_grad:
                    if len(param.shape) == 1 or name.endswith(".bias"):
                        no_wd.append(param)
                    else:
                        use_wd.append(param)

            # Add parameters with weight-decay
            optimizer_param_groups.append(
                {
                    "params": use_wd,
                    "lr": base_lr * fine_module["lr_scale"],
                    "weight_decay": self.weight_decay,
                }
            )

            # Add parameters without weight-decay
            optimizer_param_groups.append(
                {
                    "params": no_wd,
                    "lr": base_lr * fine_module["lr_scale"],
                    "weight_decay": 0.0,
                }
            )

            # build a set of parameters handled, remaining will be handled next
            finetune_params_set.update(list(fine_module["module"].parameters()))

        # remaining_params are those parameters w/ default lr
        use_wd, no_wd = [], []
        for name, param in self.named_parameters():
            if param in finetune_params_set:
                continue
            if param.requires_grad:
                if len(param.shape) == 1 or name.endswith(".bias"):
                    no_wd.append(param)
                else:
                    use_wd.append(param)

        # Add parameters with weight-decay
        optimizer_param_groups.append(
            {"params": use_wd, "weight_decay": self.weight_decay}
        )
        # Add parameters without weight-decay
        optimizer_param_groups.append({"params": no_wd, "weight_decay": 0.0})

        return optimizer_param_groups


class TextBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self, batch_dict):

        encoder_inputs = self.embeddings(batch_dict["question_indices"])
        attention_mask = batch_dict["question_mask"]

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs, extended_attention_mask, head_mask=head_mask
        )
        seq_output = encoder_outputs[0]

        return seq_output
###############
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(30522, config.hidden_size, padding_idx=0)   ##config.vocab_size:30522
        self.position_embeddings = nn.Embedding(512, config.hidden_size)   ##config.max_position_embeddings
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
        self.register_buffer("position_ids", torch.arange(512).expand((1, -1)))

    #def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
    #def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, adv_training=None, adv_modality=False, adv_delta=None):
    def forward(self, input_ids=None, inputs_embeds=None, adv_training=None, adv_modality=False, adv_delta=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        #if position_ids is None:
        position_ids = self.position_ids[:, :seq_length]

        #if token_type_ids is None:
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if adv_training == True and 'text' in adv_modality:
            inputs_embeds = inputs_embeds + adv_delta
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertImageEmbeddings(nn.Module):
    """Construct the embeddings from image, spatial location (omit now) and token_type embeddings."""

    def __init__(self, config):
        super(BertImageEmbeddings, self).__init__()
        self.image_embeddings = nn.Linear(2048, config.hidden_size)   ##self.img_linear
        self.image_location_embeddings = nn.Linear(5, config.hidden_size)   ## self.pos_linear
        self.image_type_embeddings = nn.Embedding(1, config.hidden_size)
        self.mask_embedding = nn.Embedding(2, 2048, padding_idx=0)     ##self.mask_embedding.weight.data[0, :].fill_(0)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)    ##self.img_layer_norm    ##self.final_layer_norm = LayerNorm(self.embed_dim)
        ##self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        #self.activation_fn = 
        self.dropout = nn.Dropout(config.hidden_dropout_prob)     ##0.1

    def forward(self, img_feat, img_pos_feat, adv_training, adv_modality, adv_delta, img_masks=None):   #forward(self, input_ids, input_loc):
        if img_masks is not None:
            self.mask_embedding.weight.data[0, :].fill_(0)
            mask = self.mask_embedding(img_masks.long())
            img_feat = img_feat + mask
        
        if adv_training == True and 'image' in adv_modality:
            img_feat = img_feat + adv_delta
        img_embeddings = self.image_embeddings(img_feat)     ##transformed_im = self.img_layer_norm(self.img_linear(img_feat))
        loc_embeddings = self.image_location_embeddings(img_pos_feat)
        type_ids = img_feat.new_zeros(img_embeddings.shape[:-1], dtype=torch.long)
        type_embeddings = self.image_type_embeddings(type_ids)
        embeddings = self.LayerNorm(
            img_embeddings + loc_embeddings + type_embeddings
        )
        
        embeddings = self.dropout(embeddings)
        return embeddings

class ADVContrastiveProjection(nn.Module):
    def __init__(self, config):
        super(ADVContrastiveProjection, self).__init__()

        self.tau = config.tau
        self.pos_eps = config.pos_eps
        self.neg_eps = config.neg_eps 

        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)        #768
        self.linear2 = nn.Linear(config.hidden_size, config.contrast_out_dim)   #128 
        self.two_norm = registry.two_norm
        logger.info(f"Two norm for projection is {self.two_norm}")

        self.t5_model = T5ForConditionalGeneration.from_pretrained(config.t5_model)    
        self.cont_model = MMT_VQA(self.mmt_config) 
        #self.text_bert = TextBert.from_pretrained("bert-base-uncased", config=self.text_bert_config)
        
        self.projection = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),    ##class
                                        nn.ReLU())
    #def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask,  lm_labels, adv=False):
    def forward(self, batch_dict):
        ##batch_dict["contrastive_projection_norm"] = F.normalize(                   
        ##    self.linear2(F.relu(self.linear1(batch_dict["pooled_output"]))), dim=-1)
        encoder = self.cont_model.self.encoder()

        text_embeddings = batch_dict["text_bert_emb"]
        joint_mask = torch.cat([batch_dict["question_mask"], batch_dict["image_mask"]], dim=-1)
        extended_attention_mask = joint_mask.unsqueeze(1).unsqueeze(2)  
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0  
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers
        
        embedding_output = torch.cat([self.text_embeddings(batch_dict["question_indices"], text_embeddings), 
                                      self.image_embeddings(batch_dict["input_imgs"], batch_dict["image_loc"])], dim=1)
        
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, head_mask=head_mask)  
 
        text_len = text_embeddings.shape[1]
        pooled_text_output = self.text_pooler(encoder_outputs[0][:, :text_len])
        pooled_image_output = self.image_pooler(encoder_outputs[0][:, text_len:])    ##encoder

        batch_dict["pooled_output"] = self.dropout(batch_dict["pooled_text_output"] * batch_dict["pooled_image_output"])
        
        batch_dict["vil_prediction"] = self.vil_prediction(batch_dict["pooled_output"]) 

        self.vil_prediction = SimpleClassifier(self.mmt_config.hidden_size, self.mmt_config.hidden_size * 2, 3129, 0.5)  ##encoder

        batch_dict["contrastive_projection_norm"] = F.normalize(                    
                self.linear2(F.relu(self.linear1(batch_dict["pooled_output"]))), dim=-1)

        batch_dict["contrastive_projection_norm"] = F.normalize(                    
                self.linear2(F.relu(self.linear1(batch_dict["vil_prediction"]))), dim=-1)

        hidden_states = encoder_outputs[0]

        # input_ids: ids of article tokens
        # attention_mask: mask for input_ids 0 for PAD 1 o.w
        # decoder_input_ids: ids of summary tokens
        # decoder_attention_mask: mask for decoder_input_ids 0 for PAD 1 o.w
        # lm_labels: shift decoder_input_ids left

        encoder_outputs = encoder(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  inputs_embeds=None,
                                  head_mask=None
                                  )
        #self.image_pooler(encoder_outputs[0][:, text_len:])
        hidden_states = encoder_outputs[0]    ###seq_output  textbert   self.text_bert  hidden_states = TextBert() batch_dict["text_bert_emb"]
        hidden_states = TextBert(self.text_bert_config)

        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=None,
            past_key_value_states=None,
            encoder_hidden_states=hidden_states,    ##batch_dict["text_bert_emb"]
            encoder_attention_mask=attention_mask,
            head_mask=None,
            use_cache=None,
        )
        sequence_output = decoder_outputs[0]    ##mmt_results
        # Rescale output before projecting on vocab
        sequence_output = sequence_output * (self.t5_model.model_dim ** -0.5)
        lm_logits = self.t5_model.lm_head(sequence_output)    ##batch_dict["vil_prediction"]  nn.linear

        # Add hidden states and attention if they are here
        decoder_outputs = (lm_logits,) + decoder_outputs[1:]  

        vocab_size = lm_logits.size(-1)

        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        nll = criterion(lm_logits.view(-1, vocab_size),
                        lm_labels.view(-1))
        if adv:
            proj_enc_h = F.normalize(self.linear2(F.relu(self.linear1(batch_dict["pooled_output"]))), dim=-1)         ##self.projection(hidden_states) 
            proj_dec_h = F.normalize(self.linear2(F.relu(self.linear1(batch_dict["vil_prediction"]))), dim=-1)        ##self.projection(sequence_output)
            #avg_doc = self.avg_pool(proj_enc_h, attention_mask)
            #avg_abs = self.avg_pool(proj_dec_h, decoder_attention_mask)
            
            cos = nn.CosineSimilarity(dim=-1)
            cont_crit = nn.CrossEntropyLoss()
            sim_matrix = cos(proj_enc_h.unsqueeze(1),    #sim input and out 
                             proj_dec_h.unsqueeze(0))
            #perturbed_dec = self.generate_adv(sequence_output, lm_labels)  # [n,b,t,d] or [b,t,d]
            perturbed_dec = self.generate_adv(batch_dict["vil_prediction"], batch_dict['target'])
            batch_size = input_ids.size(0)

            proj_pert_dec_h = self.projection(perturbed_dec)
            #avg_pert = self.avg_pool(proj_pert_dec_h, decoder_attention_mask)

            adv_sim = cos(proj_enc_h, proj_pert_dec_h).unsqueeze(1)  # [b,1]   #sim input and out

            pos_dec_hidden = self.generate_cont_adv(hidden_states, attention_mask,                       ##pooled_text_output   
                                                    sequence_output, decoder_attention_mask,
                                                    lm_logits,
                                                    self.tau, self.pos_eps)
            pos_dec_hidden = self.generate_cont_adv(batch_dict["pooled_output"], batch_dict["vil_prediction"])

            avg_pos_dec = self.avg_pool(self.projection(pos_dec_hidden), decoder_attention_mask)

            pos_sim = cos(proj_enc_h, pos_dec_hidden).unsqueeze(-1)  # [b,1] #sim input and out
            logits = torch.cat([sim_matrix, adv_sim], 1) / self.tau

            identity = torch.eye(batch_size, device=input_ids.device)
            pos_sim = identity * pos_sim
            neg_sim = sim_matrix.masked_fill(identity == 1, 0)
            new_sim_matrix = pos_sim + neg_sim
            new_logits = torch.cat([new_sim_matrix, adv_sim], 1)   ##pos+neg, 

            labels = torch.arange(batch_size,                             ##labels = batch_dict[0]["target"].argmax(dim=-1).view(-1, 1)
                                  device=input_ids.device)

            cont_loss = cont_crit(logits, labels)
            new_cont_loss = cont_crit(new_logits, labels)

            cont_loss = 0.5 * (cont_loss + new_cont_loss)

            return nll, cont_loss

        else:
            return nll


class ContrastiveProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear2 = nn.Linear(config.hidden_size, config.contrast_out_dim)   #128 
        self.two_norm = registry.two_norm
        logger.info(f"Two norm for projection is {self.two_norm}")

    def forward(self, batch_dict):

        if self.two_norm:   ##fasle
            # use norm twice
            pooled_norm = F.normalize(batch_dict["pooled_output"], dim=-1)
            batch_dict["contrastive_projection_norm"] = F.normalize(
                self.linear2(F.relu(self.linear1(pooled_norm))), dim=-1
            )
        else:
            batch_dict["contrastive_projection_norm"] = F.normalize(                    ##convolution layer, a batch normalization layer, relu
                self.linear2(F.relu(self.linear1(batch_dict["pooled_output"]))), dim=-1
            )


class MMT_VQA(BertPreTrainedModel):      ##uniterModel   
    def __init__(self, config):
        super().__init__(config)
        #w_emb, q_emb, att, img_emb
        self.image_embeddings = BertImageEmbeddings(config)
        self.text_embeddings = BertEmbeddings(config)
        # Add this when TextBERT is identity function,
        self.encoder = BertEncoder(config)
        self.text_pooler = BertTextPooler(config)
        self.image_pooler = BertImagePooler(config)

        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def _compute_txt_embeddings(self, input_ids, text_embeddings,
                                adv_training, adv_modality, 
                                adv_delta, txt_type_ids=None):
        output = self.embeddings(input_ids, txt_emb, 
            adv_training, adv_modality, adv_delta, txt_type_ids)
        return output

    def _compute_img_embeddings(self, img_feat, img_pos_feat,adv_training, adv_modality, 
                                adv_delta, img_masks=None, img_type_ids=None):
        if img_type_ids is None:
            img_type_ids = torch.ones_like(img_feat[:, :, 0].long()) 
        type_embeddings = self.text_embeddings.token_type_embeddings(
            img_type_ids)    
        #att = self.v_att(v, q_emb)       
        #output = self.image_embeddings(img_feat, img_pos_feat, type_embeddings, adv_training,adv_modality, adv_delta, img_masks)
        output = self.image_embeddings(img_feat, img_pos_feat, adv_training,adv_modality, adv_delta, img_masks)
        return output
    def _compute_img_txt_embeddings(self, input_ids, text_embeddings,
                                    img_feat, img_pos_feat,
                                    gather_index, adv_training, adv_modality, 
                                    adv_delta_txt, adv_delta_img, img_masks=None,
                                    txt_type_ids=None, img_type_ids=None):
        txt_emb = self._compute_txt_embeddings(
            input_ids, text_embeddings, adv_training, adv_modality, 
            adv_delta_txt, txt_type_ids)
        img_emb = self._compute_img_embeddings(
            img_feat, img_pos_feat, adv_training, adv_modality, 
            adv_delta_img, img_masks, img_type_ids)
        # align back to most compact input
        #gather_index = gather_index.unsqueeze(-1).expand(
            #-1, -1, self.config.hidden_size)
        embedding_output = torch.gather(torch.cat([txt_emb, img_emb], dim=1),
                                        dim=1)
        return embedding_output

    def forward(self, batch_dict,                        
                adv_training, adv_modality, adv_delta_txt,
                adv_delta_img, img_masks=None, img_type_ids=None):

        text_embeddings = batch_dict["text_bert_emb"]
        # text_embeddings = self.text_embeddings(batch_dict["question_indices"])
        joint_mask = torch.cat([batch_dict["question_mask"], batch_dict["image_mask"]], dim=-1)
        extended_attention_mask = joint_mask.unsqueeze(1).unsqueeze(2)  
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0  
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers
        #image_embeddings = BertAttention(config)
        embedding_output = torch.cat([self.text_embeddings(batch_dict["question_indices"], text_embeddings, adv_training, adv_modality, adv_delta_txt), 
                                      self.image_embeddings(batch_dict["input_imgs"], batch_dict["image_loc"],adv_training,adv_modality, adv_delta_img)], dim=1)

        #embedding_output = self._compute_img_txt_embeddings(batch_dict["question_indices"], text_embeddings,batch_dict["input_imgs"],batch_dict["image_loc"],
        #                                                    adv_training, adv_modality, adv_delta_txt, adv_delta_img, img_masks=None,
        #                                                    txt_type_ids=None, img_type_ids=None)
        
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, head_mask=head_mask)   ##ncoded_layers
        ##extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  
 
        text_len = text_embeddings.shape[1]
        pooled_text_output = self.text_pooler(encoder_outputs[0][:, :text_len])
        pooled_image_output = self.image_pooler(encoder_outputs[0][:, text_len:])

        results = {
            "pooled_text_output": pooled_text_output,
            "pooled_image_output": pooled_image_output,
            # "attention_weights": encoder_outputs[1] if registry.squint_loss else None
        }
        return results

class BertTextPooler(nn.Module):
    def __init__(self, config):
        super(BertTextPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertImagePooler(nn.Module):
    def __init__(self, config):
        super(BertImagePooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def _get_mask(nums, max_num):
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask


@functools.lru_cache(maxsize=32)
def _get_causal_mask(seq_length, device):
    # generate a lower triangular mask
    mask = torch.zeros(seq_length, seq_length, device=device)
    for i in range(seq_length):
        for j in range(i + 1):
            mask[i, j] = 1.0
    return mask


def _batch_gather(x, inds):
    assert x.dim() == 3
    batch_size = x.size(0)
    length = x.size(1)
    dim = x.size(2)
    x_flat = x.view(batch_size * length, dim)

    batch_offsets = torch.arange(batch_size, device=inds.device) * length
    batch_offsets = batch_offsets.unsqueeze(-1)
    assert batch_offsets.dim() == inds.dim()
    inds_flat = batch_offsets + inds
    results = F.embedding(inds_flat, x_flat)
    return results


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super().__init__()
        self.logit_fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            GeLU(),
            BertLayerNorm(hid_dim, eps=1e-12),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, hidden_states):
        return self.logit_fc(hidden_states)


class LinearClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features.detach())
