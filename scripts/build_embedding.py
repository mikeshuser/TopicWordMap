# -*- coding: utf-8 -*-
"""
Embed pre-processed texts in a vector space using FastText approach. FastText
will not produce contextually varying vectors(like Transformer architectures), 
but since the ultimate goal is a data visualization in 2d, embedding with the
average context is sufficient.

Steps:
    -Run a bigram pass on both positve and negative corpuses
    -Bigrammed corpuses into single global corpus
    -Compute FastText word vectors on global corpus

Dependencies:
    pandas == 0.23
    gensim == 3.8
"""

__author__ = "Mike Shuser"

import pickle
from pathlib import Path
import pandas as pd
from gensim.models import Phrases
from gensim.models.fasttext import FastText

DATA_SRC = "../processed_corpus" #processed data directory
SAVE_DIR = "../modelling"

def make_bigrams(
    docs_file_path: str,
    debug: bool = False,
    **phraser_args
):

    """
    Extract bigrams from a corpus
    """
    
    with open(docs_file_path) as f:
        docs = [row.split() for row in f.readlines()]
 
    print("extracting bigrams")
    bigram_model = Phrases(docs, **phraser_args)
    print("extraction complete")

    if debug:
        bigram_scores = []
        for pair in bigram_model.export_phrases(docs):
            bigram_scores.append(pair)
            bigram_scores = list(set(bigram_scores))
        print(bigram_scores)
    
    return [bigram_model[doc] for doc in docs]

def sanity_check(func, *args, **kwargs):
    print(f"Testing {func.__name__}")
    print(*args, kwargs)
    print(func(*args, **kwargs))
    print()

if __name__ == '__main__':

    #kwargs to pass to gensim model instances. See gensim docs for details
    phraser_args = dict(
        min_count=20,
        threshold=10,
    )
    fasttext_args = dict(
        size=30,
        window=5,
        min_count=10,
        sg=1,
        workers=10,
    )

    merged_docs = []
    files = [f"{DATA_SRC}/positive_text.csv", f"{DATA_SRC}/negative_text.csv"]

    for file in files:
        bigram_docs = make_bigrams(file, **phraser_args)
        
        with open(f"{file}.bigrams.pkl", "wb") as handle:
            pickle.dump(bigram_docs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        merged_docs.extend(bigram_docs)

    print("training FastText model")
    ft_model = FastText(**fasttext_args)
    ft_model.build_vocab(sentences=merged_docs)
    ft_model.train(sentences=merged_docs, 
        total_examples=len(merged_docs), 
        epochs=40)

    #quick sanity checks on the model
    sanity_check(ft_model.wv.most_similar, 'comedy', **dict(topn=5))
    sanity_check(ft_model.wv.most_similar, 'horror', **dict(topn=5))
    sanity_check(ft_model.wv.most_similar, 'boring', **dict(topn=5))
    sanity_check(ft_model.wv.most_similar, 'script', **dict(topn=5))
    sanity_check(ft_model.wv.most_similar, 'kubrick', **dict(topn=5))
    sanity_check(ft_model.wv.most_similar, 'brad_pitt', **dict(topn=5))
    sanity_check(ft_model.wv.similarity, 'western', 'john_wayne')
    
    #save the model and model vocab
    Path(SAVE_DIR).mkdir(parents=False, exist_ok=True)
    ft_model.save(f"{SAVE_DIR}/imdb_ft.model")
    w2v = dict(zip(ft_model.wv.index2word, ft_model.wv.vectors))
    w2v = pd.DataFrame(w2v).T
    w2v.to_csv(f"{SAVE_DIR}/imdb_wordvectors.csv")
