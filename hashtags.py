import pandas as pd
import re
import numpy as np
from webweb import Web
import hashlib
import networkx as nx
import matplotlib.pyplot as plt

def get_hashtags(filename):
    df = pd.read_csv(filename, sep=';', error_bad_lines=False)
    stopwords = ['#citizenscience', 'citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana']
    dfHashtagsRT = df[['Usuario', 'Texto', 'Fecha']].copy()
    dfHashtagsRT = dfHashtagsRT.drop([78202], axis=0)
    dfHashtagsRT = dfHashtagsRT.dropna()
    dfHashtagsRT = dfHashtagsRT[dfHashtagsRT['Texto'].str.match('RT:')]
    listHashtagsRT = dfHashtagsRT['Texto'].to_numpy()
    return listHashtagsRT

def get_edgesHashtagRT(values):
    edges = []
    for row in values:
        match = re.findall('#(\w+)', row)
        for hashtag in match:
            edges.append(hashtag)

def prepare_hashtags(list):
    stopwords = ['#citizenscience', 'citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana']
    list = [x.lower() for x in list]
    list = [word for word in list if word not in stopwords]
    list = np.unique(list, return_counts=True)
    list = sorted((zip(list[1], list[0])), reverse=True)
    sortedNumberHashtags, sortedHashtagsRT = zip(*list)
    return sortedNumberHashtags,sortedHashtagsRT


if __name__ == "__main__":
    print("Soy main")
else:
    print("Soy modulo")