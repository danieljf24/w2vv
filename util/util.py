import json
import numpy as np

def check_img_list(sentid, imgid):
    true_imgid = sentid.split('#')[0]
    if true_imgid.endswith('.jpg') or true_imgid.endswith('.mp4'):
        true_imgid = true_imgid[:-4]
    if  true_imgid== imgid:
        return 1
    else:
        return 0


def readSentsInfo(inputfile):
    sent_ids = []
    sents = []
    id2sents = {}
    for line in open(inputfile):
        data = line.strip().split(' ', 1)
        sent_ids.append(data[0])
        sents.append(data[1])
        id2sents[data[0]] = data[1]

    return (sent_ids, sents, id2sents)


def readImgSents(inputfile):
    sent_ids = []
    img_list = []
    sents = []
    for line in open(inputfile):
        data = line.strip().split(' ', 1)
        sent_ids.append(data[0])
        sents.append(data[1])
        img = data[0].split("#")[0].strip().split('.jpg')[0]
        if img not in img_list:
            img_list.append(img)
        else:
            assert img_list[-1] == img
    return img_list, sent_ids, sents

