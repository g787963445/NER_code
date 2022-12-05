import random
import json
random.seed(2020)

def read_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = []
        for line in f:
            if "target" in filename:
                line = line.replace("_", "-")
            data.append(line.strip().split())
    return data


def write_file(sens,zixing, labels, file_name):
    assert len(sens)==len(labels)==len(zixing)
    with open(file_name, "w", encoding="utf-8") as f:
        for i in range(len(sens)):
            for j in range(len(sens[i])):
                f.write(sens[i][j]+"    "+zixing[i][j]+"    "+labels[i][j]+"\n")
            f.write("\n")
    print(file_name + "'s datasize is " , len(sens))


def zixingandid(data_path):
    zixingtoid = {
        "[PAD]": 0,
        "[UNK]": 1
    }
    idtozixing = {
        0:"[PAD]",
        1:"[UNK]"
    }
    f = open(data_path,'r',encoding='utf-8')
    num = 2
    for line in f:
        zixingtoid[line.strip("\n")] = num
        idtozixing[num] = line.strip("\n")
        num += 1
    return zixingtoid,idtozixing

def wordandid(data_path):
    wordtoid = {
        "[PAD]": 0,
        "[UNK]": 1
    }
    idtoword = {
        0: "[PAD]",
        1: "[UNK]"
    }
    f = open(data_path, 'r', encoding='utf-8')
    num = 2
    for line in f:
        wordtoid[line.strip("\n")] = num
        idtoword[num] = line.strip("\n")
        num += 1

    return wordtoid, idtoword


def get_dict(sents, filter_word_num):
    word_count = {}
    for sent in sents:
        for word in sent:
            word_count[word] = word_count.get(word, 0) + 1

    # 过滤低频词
    word2id = {
        "[PAD]": 0,
        "[UNK]": 1
    }
    for word, count in word_count.items():
        if count >= filter_word_num:
            word2id[word] = len(word2id)

    print("Total %d tokens, filter count<%d tokens, save %d tokens."%(len(word_count)+2, filter_word_num, len(word2id)))

    return word2id, word_count

if __name__ == "__main__":
    sen_file = "datas/renminribao2014/source_BIO_2014_cropus.txt"
    label_file = "datas/renminribao2014/target_BIO_2014_cropus.txt"
    zixing_file = "datas/zixing.txt"
    sens = read_file(sen_file)
    zixing = read_file(zixing_file)
    labels = read_file(label_file)
    # get dicts
    zixingtoid,idtozixing = zixingandid(zixing_file)
    # word2id, _ = get_dict(sens, filter_word_num=5)
    wordtoid,idtoword = wordandid(sen_file)
    with open("datas/word2id.json", "w", encoding="utf-8") as f:
        json.dump(wordtoid, f, ensure_ascii=False)
        #编码，用于将dict类型的数据转成str类型，并写入到json文件。
    # shuffle
    data = list(zip(sens, zixing,labels))
    random.shuffle(data)
    sens,zixing,labels = zip(*data)

    dev_length = int(len(sens)*0.1)

    # write_file(sens[:1000], labels[:1000], "datas/dev.txt")
    # write_file(sens[1000:2000], labels[1000:2000], "datas/test.txt")
    # write_file(sens[10000:30000], labels[10000:30000], "datas/train.txt")

    write_file(sens[:dev_length],zixing[:dev_length],labels[:dev_length], "datas/dev.txt")
    write_file(sens[dev_length:2*dev_length],zixing[dev_length:2*dev_length],labels[dev_length:2*dev_length],
               "datas/test.txt")
    write_file(sens[2*dev_length:],zixing[2*dev_length:], labels[2*dev_length:], "datas/train.txt")
    