import os
from collections import Counter
import pickle as pk
import codecs

"""USED"""
def ifnodirmkdir_elsewarn(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)
    else:
        print(f'{dir} already exists. You may want to delete that folder first.')



"""NOT USED"""
def countdict(labels,totaldict):
    cdict = Counter(labels)
    for key in cdict.keys():
        if key in totaldict:
            totaldict[key] += cdict[key]
        else:
            totaldict[key] = cdict[key]
    return totaldict


def dict_save(dict_address, record_dict):
    with open(dict_address, 'wb') as wp:
        pk.dump(record_dict,wp)


def load_dict(dict_path):
    with open(dict_path, 'rb') as rp:
        dict = pk.load(rp)
    return dict

def loadlines_utf8or16(targetfile):
    try:
        with codecs.open(targetfile, 'r', encoding='utf-8') as rf:
            lines = rf.readlines()
    except:
        with codecs.open(targetfile, 'r', encoding='utf-16') as rf:
            lines = rf.readlines()
    return lines

def load_wordlist(wordlist_path):
    wordlist = []
    with open(wordlist_path) as file:
        for line in file:
            tokens = line.split()
            if tokens:
                wordlist.append(tokens[1])
    return wordlist

