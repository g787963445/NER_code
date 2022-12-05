import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from hanzi_chaizi_ import HanziChaizi


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


file_name = "datas/dev.txt"
x,y,z = read_file(file_name)

def zixingtoid(z):
    num = 1
    zixing2id = {}
    id2zixing = {}
    for i, sequence in tqdm(enumerate(z)):


        for line in sequence:

            lines = line.strip()
            zixing2id[lines] = num
            id2zixing[num] = lines
            num += 1
    return zixing2id,id2zixing

zixing2id = zixingtoid(z)