# -*- coding: utf-8 -*-
"""
Project word vectors into 2d space with UMAP, then search for clusters

Dependencies:
    pandas == 0.23
    umap == 0.5.1
    sklearn == 0.24.2
    seaborn == 0.11
"""

__author__ = "Mike Shuser"

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import AgglomerativeClustering

MODEL_DIR = "../modelling"
VOCAB_THRESH = 100
N_NEIGHBORS = 6
MIN_DIST = 0.1
DIST_THRESH = 0.9
DEBUG = True

def project(
    vec_space: pd.DataFrame,
    debug: bool = False,
    **umap_args
) -> np.ndarray:

    """
    Project vector space into 2 dimensions
    """

    umap_space = umap.UMAP(**umap_args).fit_transform(vec_space.to_numpy())
    if debug:
        print('showing projection scatterplot')
        sns.scatterplot(x=umap_space[:,0], y=umap_space[:,1], marker=".")
        plt.show()

    return umap_space

def cluster(
    umap_space: np.ndarray,
    debug: bool = False,
    **cluster_args
) -> list:

    """
    Group the 2d map into clusters
    """

    clustering = AgglomerativeClustering(**cluster_args).fit(umap_space)
    labels = list(clustering.labels_)
    palette = np.array(sns.color_palette(n_colors=len(set(labels))))

    if debug:
        print('showing cluster scatterplot')
        sns.scatterplot(x=umap_space[:,0], y=umap_space[:,1], 
            s=10, 
            c=palette[labels])
        plt.show()

    return labels

def most_frequent_words(vocab_lists: list, threshold: int) -> pd.Series:

    """
    Return vocabulary that contains mentions >= theshold in at least one of
    the applicable corpuses
    """

    init = False
    for vocab in vocab_lists:
        assert isinstance(vocab, pd.Series), \
            "vocab_lists must contain pandas Series"
        
        if not init:
            keep_words = pd.Series(index=vocab.index, data=False)
            init = True

        keep_words = (vocab >= threshold) | keep_words

    return keep_words

if __name__ == '__main__':

    vec_space = pd.read_csv(f"{MODEL_DIR}/imdb_wordvectors.csv", 
        index_col=[0],
        na_filter=False)

    sentiment = ["positive","negative"]
    vocabs = [pd.read_csv(f"{MODEL_DIR}/{sent}_text_vocab_stats.csv",
        index_col=0,
        na_filter=False)["mentions"] for sent in sentiment]
    final_vocab = most_frequent_words(vocabs, VOCAB_THRESH)
    vec_space = vec_space.loc[final_vocab, :]
    print(f"Vocabulary trimmed to {len(vec_space)} most frequent words")

    umap_args = dict(
        n_components=2,
        metric='cosine',
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
    )

    cluster_args = dict(
        linkage='average',
        n_clusters=None,
        distance_threshold=DIST_THRESH,
    )

    umap_space = project(vec_space, DEBUG, **umap_args)
    umap_data = pd.DataFrame(data=umap_space, index=vec_space.index)
    umap_data.reset_index(inplace=True)
    umap_data.columns = ['word', 'x', 'y']

    cluster_labels = cluster(umap_space, DEBUG, **cluster_args)
    umap_data['clusters'] = cluster_labels

    umap_data.to_csv(f"{MODEL_DIR}/umap_projection.csv", index=True)




        
