import nltk
import os
import math
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import glob
import re
from nltk.util import ngrams
import utils
from collections import *
import random


def concat_speeches(president:str, order=4):
    fpath = "../data/" + president +"/"
    all_speeches = ""
    for file in os.listdir(fpath):
        s = open(fpath+file, "r").read()
        all_speeches += s
    return all_speeches


def train_char_lm_on_single_speech(fname, order=4):
    data = open(fname, "r").read()
    lm = defaultdict(Counter)
    pad = "~" * order
    data = pad + data
    for i in range(len(data)-order):
        history, char = data[i:i+order], data[i+order]
        lm[history][char]+=1

    def normalize(counter):
        s = float(sum(counter.values()))
        return [(c,cnt/s) for c,cnt in counter.items()]

    outlm = {hist:normalize(chars) for hist, chars in lm.items()}
    return outlm

def train_char_lm(president="reagan", order=4):
    data = concat_speeches(president, order)
    lm = defaultdict(Counter)
    pad = "~" * order
    data = pad + data
    for i in range(len(data)-order):
        history, char = data[i:i+order], data[i+order]
        lm[history][char]+=1

    def normalize(counter):
        s = float(sum(counter.values()))
        return [(c,cnt/s) for c,cnt in counter.items()]

    outlm = {hist:normalize(chars) for hist, chars in lm.items()}
    return outlm


def generate_letter(lm, history, order):
    history = history[-order:]
    dist = lm[history]
    x = random.random()
    for c,v in dist:
        x = x - v
        if x <= 0: return c


def generate_text(lm, order, nletters=1000):
    history = "~" * order
    out = []
    for i in range(nletters):
        c = generate_letter(lm, history, order)
        history = history[-order:] + c
        out.append(c)
    return "".join(out)
