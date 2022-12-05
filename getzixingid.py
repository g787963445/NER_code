from hanzi_chaizi_ import HanziChaizi

def is_Chinese(ch):

    if '\u4e00' <= ch <= '\u9fff':
        return True
    return False

hc = HanziChaizi()
def get_zixing_ids():
    f = open("datas/source.txt","r",encoding='utf-8')
    zixing2id = {}
    id2zixing = {}
    zixings = []
    num = 1
    for line in f:
        word = line.strip()
        if is_Chinese(word):
            try:
                zixing = hc.query(word)
                # f.write(zixing)
                zixings.append(zixing)

            except TypeError:
                zixings.append(word)
                # f.write(word)
        else:
            zixings.append(word)

    zixing_id = []

    for i in zixings:
        for k in i:
            zixing2id[k] = num
            id2zixing[num] = k
            num+=1

    for k,v in zixing2id.items():
        zixing_id.append(v)

    return zixing_id
def get_zixing_id():
    f = open("datas/source.txt","r",encoding='utf-8')
    zixing2id = {}
    id2zixing = {}
    zixings = []
    num = 1
    for line in f:
        word = line.strip()
        if is_Chinese(word):
            try:
                zixing = hc.query(word)
                # f.write(zixing)
                zixings.append(zixing)

            except TypeError:
                zixings.append(word)
                # f.write(word)
        else:
            zixings.append(word)

    zixing_id = []

    for i in zixings:
        for k in i:
            zixing2id[k] = num
            id2zixing[num] = k
            num+=1


    return zixing2id,id2zixing
