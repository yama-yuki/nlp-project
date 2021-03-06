'''
coding BERT from scratch with 「つくりながら学ぶ！PyTorchによる発展ディープラーニング」ch.8
'''

import json
import math
import os
from attrdict import AttrDict

import torch
from torch import nn
from torch.nn import functional as F

from transformers import BertConfig
config = BertConfig.from_pretrained('bert-base-uncased')

class BertLayerNorm(nn.Module):
    ## LayerNormalization

    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()

        self.gamma = nn.Parameter(torch.ones(hidden_size)) ## weight
        self.beta = nn.Parameter(torch.zeros(hidden_size)) ## bias
        self.variance_epsilon = eps

    def forward(self, x):

        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)

        return self.gamma * x + self.beta

class BertEmbeddings(nn.Module):

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()

        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)       
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)
        
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        ## input_ids：list of word ids
        ## token_type_ids：list of snt1/snt2

        words_embeddings = self.word_embeddings(input_ids)
        ## if only snt1
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        ## snt1 or snt1/snt2
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class BertLayer(nn.Module):
    ## Transformer BertLayer

    def __init__(self, config):
        super(BertLayer, self).__init__()

        ## Self-Attention
        self.attention = BertAttention(config)
        ## Fully Connected Layer
        self.intermediate = BertIntermediate(config)
        ## Self-Attention feat + input to BertLayer
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, attention_show_flg=False):
        ## hidden_states：tensor[batch_size, seq_len, hidden_size] from Embeddings
        ## attention_mask：Transformer mask
        ## attention_show_flg：whether to return Self-Attention weight

        if attention_show_flg == True:
            ## if attention_show, return attention_probs
            attention_output, attention_probs = self.attention(
                hidden_states, attention_mask, attention_show_flg)
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
            return layer_output, attention_probs

        elif attention_show_flg == False:
            attention_output = self.attention(
                hidden_states, attention_mask, attention_show_flg)
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)

            return layer_output  # [batch_size, seq_length, hidden_size]

class BertAttention(nn.Module):
    ## Self-Attention of BertLayer
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.selfattn = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, attention_show_flg=False):
        ## input_tensor：output from Embeddings or BertLayer
        ## attention_mask：Transformer mask
        ## attention_show_flg：whether to return Self-Attention weight

        if attention_show_flg == True:
            ## if attention_show, return attention_probs
            self_output, attention_probs = self.selfattn(input_tensor, attention_mask, attention_show_flg)
            attention_output = self.output(self_output, input_tensor)
            return attention_output, attention_probs
        
        elif attention_show_flg == False:
            self_output = self.selfattn(input_tensor, attention_mask, attention_show_flg)
            attention_output = self.output(self_output, input_tensor)
            return attention_output

class BertSelfAttention(nn.Module):
    ## Self-Attention of BertAttention

    def __init__(self, config):
        super(BertSelfAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads # num_attention_heads': 12

        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)  # 768/12=64
        self.all_head_size = self.num_attention_heads * \
            self.attention_head_size  # = 'hidden_size': 768

        ## FCL to make Self-Attention feat
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        ## Dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        ## convert tensor for multi-head Attention
        ## [batch_size, seq_len, hidden] to [batch_size, 12, seq_len, hidden/12] 

        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, attention_show_flg=False):
        ## input_tensor：output from Embeddings or BertLayer
        ## attention_mask：Transformer mask
        ## attention_show_flg：whether to return Self-Attention weight

        ## feat conversion in FCL for all multi-head Attention
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        ## convert tensor for multi-head Attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        ## matmul feats to get similarities as Attention_scores
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)

        ## masking
        attention_scores = attention_scores + attention_mask

        ## normalize Attention
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        ## dropout
        attention_probs = self.dropout(attention_probs)

        ## matmul Attention Map
        context_layer = torch.matmul(attention_probs, value_layer)

        ## recover multi-head Attention tensor
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        ## if attention_show, return attention_probs
        if attention_show_flg == True:
            return context_layer, attention_probs
        elif attention_show_flg == False:
            return context_layer

class BertSelfOutput(nn.Module):
    ## FCL to process BertSelfAttention outoput

    def __init__(self, config):
        super(BertSelfOutput, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob) # 'hidden_dropout_prob': 0.1

    def forward(self, hidden_states, input_tensor):
        ## hidden_states：output BertSelfAttention tensor
        ## input_tensor：output from Embeddings or BertLayer

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

def gelu(x):
    ## Gaussian Error Linear Unit Activation Function
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class BertIntermediate(nn.Module):
    ## FeedForward of TransformerBlock 
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size) # FCL：'hidden_size': 768、'intermediate_size': 3072

        self.intermediate_act_fn = gelu
            
    def forward(self, hidden_states):
        ## hidden_states：output BertAttention tensor

        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states) #gelu
        return hidden_states

class BertOutput(nn.Module):
    ## FeedForward of TransformerBlock 

    def __init__(self, config):
        super(BertOutput, self).__init__()

        ## FCL：'intermediate_size': 3072、'hidden_size': 768
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(config.hidden_dropout_prob) # 'hidden_dropout_prob': 0.1

    def forward(self, hidden_states, input_tensor):
        ## hidden_states： output tensor of BertIntermediate
        ## input_tensor：output tensor of BertAttention

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertEncoder(nn.Module):
    def __init__(self, config):
        ## repetitive BertLayer
        super(BertEncoder, self).__init__()

        ## create config.num_hidden_layers=12 BertLayer
        self.layer = nn.ModuleList([BertLayer(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, attention_show_flg=False):
        ## hidden_states：Embeddings output
        ## attention_mask：Transformer mask
        ## output_all_encoded_layers：flag to return all TransformerBlock output or just output of the final layer
        ## attention_show_flg：flag to rerturn Self-Attention weight

        all_encoder_layers = []

        ## repeat process of BertLayer
        for layer_module in self.layer:

            if attention_show_flg == True:
                ## if attention_show, return attention_probs
                hidden_states, attention_probs = layer_module(
                    hidden_states, attention_mask, attention_show_flg)
            elif attention_show_flg == False:
                hidden_states = layer_module(
                    hidden_states, attention_mask, attention_show_flg)

            ## if return all TransformerBlock output
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        ## elif return just output of the final layer
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        ## if attention_show, return attention_probs (12th)
        if attention_show_flg == True:
            return all_encoder_layers, attention_probs
        elif attention_show_flg == False:
            return all_encoder_layers

class BertPooler(nn.Module):
    ## convert feat of [cls]

    def __init__(self, config):
        super(BertPooler, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size) # FCL 'hidden_size': 768
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        ## feat of first word
        first_token_tensor = hidden_states[:, 0]

        ## feat conversion at FCL
        pooled_output = self.dense(first_token_tensor)

        ## Tanh
        pooled_output = self.activation(pooled_output)

        return pooled_output

class BertModel(nn.Module):

    def __init__(self, config):
        super(BertModel, self).__init__()

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True, attention_show_flg=False):
        ## input_ids： list of tok id
        ## token_type_ids： list of snt1/snt2
        ## attention_mask：Transformer mask
        ## output_all_encoded_layers： all 12 or only the final
        ## attention_show_flg：flag to return Self-Attention weight

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)

        if attention_show_flg == True:
            encoded_layers, attention_probs = self.encoder(embedding_output,
                                                           extended_attention_mask,
                                                           output_all_encoded_layers, attention_show_flg)

        elif attention_show_flg == False:
            encoded_layers = self.encoder(embedding_output,
                                          extended_attention_mask,
                                          output_all_encoded_layers, attention_show_flg)

        ## BertPooler
        pooled_output = self.pooler(encoded_layers[-1])

        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        if attention_show_flg == True:
            return encoded_layers, pooled_output, attention_probs
        elif attention_show_flg == False:
            return encoded_layers, pooled_output


if __name__ == '__main__':
    '''
    print(config)

    input_ids = torch.LongTensor([[31, 51, 12, 23, 99], [15, 5, 1, 0, 0]])
    attention_mask = torch.LongTensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])

    net = BertModel(config)

    encoded_layers, pooled_output, attention_probs = net(
        input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, attention_show_flg=True)

    print(encoded_layers.shape)
    print(pooled_output.shape)
    print(attention_probs.shape)
    '''

    weights_path = "../qa/models/bert-base-uncased-pytorch_model.bin"
    loaded_state_dict = torch.load(weights_path)

    net = BertModel(config)
    net.eval()

    param_names = [name for name, _ in net.named_parameters()]
    new_state_dict = net.state_dict().copy()

    for index, (key_name, value) in enumerate(loaded_state_dict.items()):
        name = param_names[index]
        new_state_dict[name] = value
        if index+1 >= len(param_names):
            break

    net.load_state_dict(new_state_dict)
