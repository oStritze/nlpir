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

# Highly inspired by https://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139

def concat_speeches(president:str, order=4):
    fpath = "../data/" + president +"/"
    all_speeches = ""
    for file in os.listdir(fpath):
        if file == ".ipynb_checkpoints":
            next
        else:
            s = open(fpath+file, "r").read()
            #s = s.replace("\n\n", "\n")
            pad = "~" * order
            all_speeches += pad + s + "\n" 
            #all_speeches += s
    return all_speeches + "~" * order # if the last speeches end is called we get an error since there is no follow up on this - therefore, start a new one...
    
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
    #pad = "~" * order
    #data = pad + data
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
    #print(history, dist, x)
    for c,v in dist:
        x = x - v
        # do not predict a "~" since this would mix different starts into one speech - this could be used for early-ending of the speech though
        #if x <= 0 and c != "~": 
        if x <= 0:
            return c

		
def generate_text(lm, order, nletters=1000, early_stopping=False):
    history = "~" * order
    out = []
    for i in range(nletters):
        c = generate_letter(lm, history, order)
        if c == None or c == "~":
            if early_stopping:
                #print(c)
                #print(history, lm[history[-order:]]) 
                break
            if not early_stopping:
                history = history + "~" * order
                c = generate_letter(lm, history, order)
            #print(history, lm[history[-order:]]) 
        history = history[-order:] + c
        out.append(c)
    outstring = "".join(out)
    ## kill last unfinished word
    cnt = len(outstring) - 1
    while cnt>=0:
        if outstring[cnt] in [" ", ",", "."]:
            break
        else:    
            cnt = cnt -1
    return outstring