# -*- coding: utf-8 -*-
"""
Compute essential stats(freq/tf-idf) on a corpus

Dependencies:
    pandas == 0.23
    gensim == 3.8
"""

__author__ = "Mike Shuser"

import pickle
import numpy as np
import pandas as pd
from gensim.models import TfidfModel
from gensim.corpora import Dictionary

DATA_SRC = "../processed_corpus"
MODEL_SRC = "../modelling"

if __name__ == '__main__':

    files = ["positive_text","negative_text"]
    vecs = pd.read_csv(f"{MODEL_SRC}/imdb_wordvectors.csv", 
        index_col=[0],
        na_filter=False)

    for filetype in files:
        with open(f"{DATA_SRC}/{filetype}.csv.bigrams.pkl", "rb") as handle:
            docs = pickle.load(handle)

        vocab = pd.DataFrame(index=vecs.index)
        dct = Dictionary(docs)
        corpus = [dct.doc2bow(line) for line in docs]
        tfidf = TfidfModel(corpus)

        #corpus statistics
        def lookup_mentions(x):
            try:
                return dct.cfs[dct.token2id[x]]
            except KeyError:
                return 0
        vocab['mentions'] = vocab.index.map(lookup_mentions)
        vocab['log2_mentions'] = np.log2(vocab.mentions)
        
        #get tf-idfs for every word in each doc, then get average per word
        vocab_tfidf = {k : [] for k in vocab.index}
        for i, row in enumerate(docs):
            tmp = dict(tfidf[corpus[i]])
            for word in row:
                if word in vocab_tfidf:
                    vocab_tfidf[word].extend([tmp[dct.token2id[word]]])
                    
        for k, v in vocab_tfidf.items():
            vocab_tfidf[k] = np.mean(v)
        vocab['avg_tfidf'] = vocab.index.map(lambda x: vocab_tfidf[x])
        
        vocab.to_csv(f"{MODEL_SRC}/{filetype}_vocab_stats.csv")
        

    
