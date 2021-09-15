# -*- coding: utf-8 -*-
"""
Text pre-processing for thematic data visualisation. 
Steps include:
    -remove html tags
    -expand/remove punctuation
    -remove numbers, stop words
    -singularize words
    -lemmatize words

Dependencies:
    pandas == 0.23
    spacy == 2.1.4
    textblob == 0.15.3
"""

__author__ = "Mike Shuser"

import math
import glob
from pathlib import Path
import multiprocessing as mp

import pandas as pd
import spacy

import text_prep as tp

DATA_SRC = "../aclImdb" #path where your source data is located
SAVE_DIR = "../processed_corpus"
CPUS = mp.cpu_count() - 1 #how many cpus to use

def process_pipeline(nlp, path_list: list):
    
    """
    Full text processing pipeline. 
    """

    #IMDB review data is not too large to fit on RAM, so I'll read all text into
    #pd.Series. Would need to adapt this to run in batches for larger corpuses
    cleaned = pd.Series(index=pd.RangeIndex(len(path_list)), dtype='string')
    for i in cleaned.index:
        with open(path_list[i]) as f:
            txt = f.readlines()[0]
    
        txt = tp.remove_html(txt)
        txt = tp.norm_apostrophe(txt)
        txt = tp.expand_contractions(txt)
        txt = tp.clean_puncts(txt)
        txt = tp.remove_non_alpha(txt)
        txt = tp.remove_stop(txt, nlp)
        txt = tp.singularize(txt, nlp)
        txt = tp.lemmatize(txt, nlp)

        cleaned[i] = txt

    return cleaned

if __name__ == '__main__':
    """glob specific folders containing data. With IMDB data, reviews are 
    located in 4 different directories. I'll be merging train/test together 
    since I'm not building a classifier, but I'll keep pos/neg separate"""
    paths = [glob.glob(f"{DATA_SRC}/*/pos/*.txt"), 
        glob.glob(f"{DATA_SRC}/*/neg/*.txt")]

    nlp = spacy.load("en_core_web_sm")
    Path(SAVE_DIR).mkdir(parents=False, exist_ok=True)

    for path_list in paths:

        #Some IMDB reviews are quite lengthy, so even though n=50,000 is not 
        #that many docs, processing takes a while with a single CPU. 
        #Multiprocessing improves this significantly.
        batch_size = math.ceil(len(path_list) / CPUS)
        mp_args = [(spacy.load("en_core_web_sm"), 
            path_list[p * batch_size : (p + 1) * batch_size]) 
            for p in range(CPUS)]
        pool = mp.Pool(processes=CPUS)
        res = pool.starmap(process_pipeline, mp_args)
        pool.close()
        pool.join()

        processed = pd.concat(res)
        if "pos" in path_list[0]:
            save_path = f"{SAVE_DIR}/positive_text.csv"
        else:
            save_path = f"{SAVE_DIR}/negative_text.csv"

        processed.to_csv(save_path, index=False, header=False)
