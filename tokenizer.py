'''
from
https://github.com/YutaroOgawa/pytorch_advanced/blob/master/8_nlp_sentiment_bert/8-2-3_bert_base.ipynb
https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization.py
'''

import collections
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from transformers import BertConfig, BasicTokenizer, WordpieceTokenizer

from bert import BertModel

def load_vocab(vocab_file):
    ## vocab: txt -> dict
    vocab = collections.OrderedDict()  # (word, id)
    ids_to_tokens = collections.OrderedDict()  # (id, word)
    index = 0

    with open(vocab_file, "r", encoding="utf-8") as f:
        while True:
            token = f.readline()
            if not token:
                break
            token = token.strip()

            vocab[token] = index
            ids_to_tokens[index] = token
            index += 1

    return vocab, ids_to_tokens

class BertTokenizer(object):
    def __init__(self, vocab_file, do_lower_case=True):

        self.vocab, self.ids_to_tokens = load_vocab(vocab_file)
        never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")

        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
                                              never_split=never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token="[UNK]")

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        return ids

    def convert_ids_to_tokens(self, ids):
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens


if __name__ == '__main__':
    vocab_file = "../qa/models/bert-base-uncased/bert-base-uncased-vocab.txt"
    vocab, ids_to_tokens = load_vocab(vocab_file)

    tokenizer = BertTokenizer(
        vocab_file=vocab_file, do_lower_case=True)

    text_1 = "[CLS] I accessed the bank account. [SEP]"
    text_2 = "[CLS] He transferred the deposit money into the bank account. [SEP]"
    text_3 = "[CLS] We play soccer at the bank of the river. [SEP]"

    tokenized_text_1 = tokenizer.tokenize(text_1)
    tokenized_text_2 = tokenizer.tokenize(text_2)
    tokenized_text_3 = tokenizer.tokenize(text_3)
    print(tokenized_text_1)

    ## word -> id
    indexed_tokens_1 = tokenizer.convert_tokens_to_ids(tokenized_text_1)
    indexed_tokens_2 = tokenizer.convert_tokens_to_ids(tokenized_text_2)
    indexed_tokens_3 = tokenizer.convert_tokens_to_ids(tokenized_text_3)

    ## POSITION of word
    bank_posi_1 = np.where(np.array(tokenized_text_1) == "bank")[0][0]  # 4
    bank_posi_2 = np.where(np.array(tokenized_text_2) == "bank")[0][0]  # 8
    bank_posi_3 = np.where(np.array(tokenized_text_3) == "bank")[0][0]  # 6

    ## seqId

    ## list -> tensor
    tokens_tensor_1 = torch.tensor([indexed_tokens_1])
    tokens_tensor_2 = torch.tensor([indexed_tokens_2])
    tokens_tensor_3 = torch.tensor([indexed_tokens_3])

    ## word id
    bank_word_id = tokenizer.convert_tokens_to_ids(["bank"])[0]
    print(tokens_tensor_1)

    config = BertConfig.from_pretrained('bert-base-uncased')

    weights_path = "../qa/models/bert-base-uncased/bert-base-uncased-pytorch_model.bin"
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

    with torch.no_grad():
        encoded_layers_1, _ = net(tokens_tensor_1, output_all_encoded_layers=True)
        encoded_layers_2, _ = net(tokens_tensor_2, output_all_encoded_layers=True)
        encoded_layers_3, _ = net(tokens_tensor_3, output_all_encoded_layers=True)

    bank_vector_0 = net.embeddings.word_embeddings.weight[bank_word_id]

    bank_vector_1_1 = encoded_layers_1[0][0, bank_posi_1]
    bank_vector_1_12 = encoded_layers_1[11][0, bank_posi_1]
    bank_vector_2_1 = encoded_layers_2[0][0, bank_posi_2]
    bank_vector_2_12 = encoded_layers_2[11][0, bank_posi_2]
    bank_vector_3_1 = encoded_layers_3[0][0, bank_posi_3]
    bank_vector_3_12 = encoded_layers_3[11][0, bank_posi_3]

