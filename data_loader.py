import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from hanzi_chaizi_ import HanziChaizi
from getzixingid import get_zixing_id


def is_Chinese(ch):

    if '\u4e00' <= ch <= '\u9fff':
        return True
    return False

hc = HanziChaizi()

def read_file(filename):
    X, y,Z = [], [],[]
    #x是字，y是标签，z是部首
    labels = []
    with open(filename, 'r', encoding='utf-8') as f:
        x0, y0,z0 = [], [],[]
        for line in f:
            data = line.strip()
            if data:
                x0.append(data.split()[0])
                z0.append(data.split()[1])
                y0.append(data.split()[-1])

            else:
                if len(x0)!=0:
                    X.append(x0)
                    y.append(y0)
                    Z.append(z0)
                x0, y0 ,z0= [], [],[]
        if len(x0)!=0:
            X.append(x0)
            y.append(y0)
            Z.append(z0)
    return X, y,Z

def encode_plus(tokenizer, sequence):
    # sequence: ["中", "国", "的", "首", "都", "是", "北", "京"]
    input_ids = []
    pred_mask = []
    # wordpiece 只取第一个sub token预测
    for word in sequence:
        sub_tokens_ids = tokenizer.encode(word, add_special_tokens=False)  #首尾添加cls和sep 先分词 然后变成对应id
        input_ids = input_ids + sub_tokens_ids
        pred_mask = pred_mask + [1] + [0 for i in range(len(sub_tokens_ids)-1)]
        
    assert len(input_ids) == len(pred_mask)
    return input_ids, pred_mask


#
# def wordstoid(x):
#     num = 1
#     word2id = {}
#     id2word = {}
#     for i, sequence in tqdm(enumerate(x)):
#
#
#         for line in sequence:
#
#             lines = line.strip()
#             word2id[lines] = num
#             id2word[num] = lines
#             num += 1
#     return word2id,id2word

def wordtozixing(sequence):
    word2zixing = {}
    zixing2word = {}

    for line in sequence:
        lines = line.strip()
        zixing = ""
        if is_Chinese(lines):
            try:
                zixings = hc.query(lines)
                for i in zixings:
                    zixing += i
                word2zixing[lines] = zixing
                zixing2word[zixing] = lines
            except TypeError:
                zixing = lines
                word2zixing[lines] = zixing
                zixing2word[zixing] = lines
        else:
            zixing = lines
            word2zixing[lines] = zixing
            zixing2word[zixing] = lines
    return  word2zixing,zixing2word

zixing2id,id2zixing = get_zixing_id()


def zixingandid(sequence):
    word2zixing, _ = wordtozixing(sequence)
    # print(word2zixing)
    zixing_ids = []
    for k, v in word2zixing.items():
        for j in v.strip("\n"):
            zixing_ids.append(zixing2id[j])

    return zixing_ids



def sequence_padding_for_bert(X, y, Z,tokenizer, labels, max_len):
    input_ids_list = []
    attention_mask_list = []
    pred_mask_list = []
    input_labels_list = []
    input_zixing_id_list = []

    cls_id = tokenizer.convert_tokens_to_ids("[CLS]")
    sep_id = tokenizer.convert_tokens_to_ids("[SEP]")
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
    
    for i, sequence in tqdm(enumerate(X)):
        # get input_ids, pred_mask
        input_ids, pred_mask = encode_plus(tokenizer, sequence)
        attention_mask = [1] * len(input_ids)

        # padding
        input_ids = [cls_id] + input_ids[:max_len-2] + [sep_id] + [pad_id]* (max_len - len(input_ids) - 2) 
        pred_mask = [0] + pred_mask[:max_len-2] + [0] + [0]* (max_len - len(pred_mask) - 2)

        #get input_zixing

        # word2zixing,zixing2word = word2zixing(sequence)
        zixing_ids = zixingandid(sequence)
        zixing_ids = [cls_id] + zixing_ids[:max_len-2] + [sep_id] + [pad_id]* (max_len - len(zixing_ids) - 2)

        
        # get attention_mask
        attention_mask = [1] + attention_mask[:max_len-2] + [1] + [0]* (max_len - len(attention_mask) - 2)

        # get input_labels
        sequence_labels = [labels.index(l) for l in y[i][:sum(pred_mask)]]
        sequence_labels = sequence_labels[::-1]
        input_labels = [sequence_labels.pop() if pred_mask[i]==1 else labels.index("O") for i in range(len(pred_mask))]


        
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        pred_mask_list.append(pred_mask)
        input_labels_list.append(input_labels)
        input_zixing_id_list.append(zixing_ids)

    return torch.LongTensor(input_ids_list), \
            torch.ByteTensor(attention_mask_list), \
            torch.ByteTensor(pred_mask_list), \
            torch.LongTensor(input_labels_list), \
            torch.LongTensor(input_zixing_id_list)

# def sequence_padding_for_bilstm(X, y,Z, word2id, labels, max_len):
#     input_ids_list = []
#     attention_mask_list = []
#     pred_mask_list = []
#     input_labels_list = []
#     for i in tqdm(range(len(X))):
#         input_ids = [word2id.get(char, word2id["[UNK]"]) for char in X[i]]
#         input_labels = [labels.index(l) for l in y[i]]
#         pred_mask = [1] * len(input_ids)
#         attention_mask = [1] * len(input_ids)
#
#         input_ids = input_ids[:max_len] + [word2id["[PAD]"]]* (max_len - len(input_ids))
#         input_labels = input_labels[:max_len] + [labels.index("O")]* (max_len - len(input_labels))
#         pred_mask = pred_mask[:max_len] + [0]* (max_len - len(pred_mask))
#         attention_mask = attention_mask[:max_len] + [0]* (max_len - len(attention_mask))
#
#         input_ids_list.append(input_ids)
#         attention_mask_list.append(attention_mask)
#         pred_mask_list.append(pred_mask)
#         input_labels_list.append(input_labels)
#
#     return torch.LongTensor(input_ids_list), \
#             torch.ByteTensor(attention_mask_list), \
#             torch.ByteTensor(pred_mask_list), \
#             torch.LongTensor(input_labels_list)

class NERDataset(Dataset):
    
    def __init__(self, file_path, labels, word2id=None, tokenizer=None, max_len=128, is_BERT=True):
        self.X, self.y,self.Z = read_file(file_path)
        if is_BERT:
            self.input_ids, self.attention_masks, self.pred_mask, self.input_labels,self.zixing_ids = sequence_padding_for_bert(self.X, self.y,self.Z, tokenizer, labels,max_len=max_len)
        # else:
        #     self.input_ids, self.attention_masks, self.pred_mask, self.input_labels = sequence_padding_for_bilstm(self.X, self.y, word2id, labels, max_len=max_len)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.pred_mask[idx], self.input_labels[idx],self.zixing_ids[idx]
