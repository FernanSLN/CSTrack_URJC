import pandas as pd
import re
import numpy as np

#Código para calcular grafos usando filtros para ICALT

def get_subgraphs(graph):
    import networkx as nx
    components = list(nx.connected_components(graph))
    list_subgraphs = []
    for component in components:
        list_subgraphs.append(graph.subgraph(component))

    return list_subgraphs

def get_cites(filename):
    df = pd.read_csv(filename, sep=';', error_bad_lines=False)
    df = df.drop([78202], axis=0)
    df = df[df['Texto'].str.contains("|".join(keywords), case=False).any(level=0)]
    dfMentions = df[['Usuario', 'Texto']].copy()
    dfMentions = dfMentions.dropna()
    dfEliminarRTs = dfMentions[dfMentions['Texto'].str.match('RT @')]
    dfMentions = dfMentions.drop(dfEliminarRTs.index)
    mentionsSubset = dfMentions[['Usuario', 'Texto']]
    mentionsList = [list(x) for x in mentionsSubset.to_numpy()]
    return mentionsList

def get_retweets(filename):
    df = pd.read_csv(filename, sep=';', error_bad_lines=False)
    df = df[df['Texto'].str.contains("|".join(keywords), case=False).any(level=0)]
    dfRT = df[['Usuario', 'Texto', 'Fecha']].copy()  # Se copia a un dataframe de trabajo
    idx = dfRT['Texto'].str.contains('RT @', na=False)
    dfRT = dfRT[idx]  # Se seleccionan sólo las filas con RT
    subset = dfRT[['Usuario', 'Texto']]  # Se descarta la fecha
    retweetEdges = [list(x) for x in subset.to_numpy()]  # Se transforma en una lista
    return retweetEdges

def get_edges(values):
    edges = []
    for row in values:
        reg = re.search('@(\w+)', row[1])
        if reg:
            matchRT = reg.group(1)  # Se extrae la primera mención que hace referencia a la cuenta retuiteada
            # row[1] = hashlib.md5(match.encode()).hexdigest()
            row[1] = matchRT  # Convierte el nombre de la cuenta en hash y lo asigna al elemento
            edges.append(row)
    return edges

def get_hashtagsRT2(filename):
    df = pd.read_csv(filename, sep=';', error_bad_lines=False)
    df = df.drop([78202], axis=0)
    df = df[df['Texto'].str.contains("|".join(keywords), case=False).any(level=0)]
    dfHashtagsRT = df[['Usuario', 'Texto']]
    idx = dfHashtagsRT['Texto'].str.match('RT @', na=False)
    dfHashtagsRT = dfHashtagsRT[idx]
    listHashtagsRT = [list(x) for x in dfHashtagsRT.to_numpy()]
    return listHashtagsRT


def get_edgesHashRT2(values):
    edges = []
    for row in values:
        match = re.search('#(\w+)', row[1])
        if match:
            matchHashRT = match.group(1)
            row[1] = matchHashRT
            edges.append(row)
    return edges

# Combinación de ls ejes de RTs y Citas:

def combined_edges(x,y):
    combined_edges = x + y
    return combined_edges

# Código para calcular grafo de hashtags main

def get_hashtagsmain(filename):
    df = pd.read_csv(filename, sep=';', error_bad_lines=False)
    df = df.drop([78202], axis=0)
    df = df[df['Texto'].str.contains("|".join(keywords), case=False).any(level=0)]
    dfMainHashtags = df[['Usuario', 'Texto']].copy()
    dfMainHashtags = dfMainHashtags.dropna()
    idx = dfMainHashtags[dfMainHashtags['Texto'].str.match('RT @')]
    dfMainHashtags = dfMainHashtags.drop(idx.index)
    subset = dfMainHashtags['Texto']
    listMainHashtags = subset.to_numpy()
    return listMainHashtags

def mainHashtags(values):
    stopwords = ['#citizenscience', 'citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana']
    mainHashtags = []
    aristasHashtags = []
    for row in values:
        match = re.findall('#(\w+)', row.lower())
        length = len(match)
    try:
        match = [word for word in match if word not in stopwords]
    except ValueError:
        pass
    for index,hashtag in enumerate(match):
        mainHashtags.append(hashtag)
        if index | (length-2):
            nextHashtags = match[index+1:length-1]
            for nextHashtags in nextHashtags:
                aristasHashtags.append([hashtag,nextHashtags])
    return aristasHashtags

def prepare_hashtags2(list):
    mainHashtags = np.unique(list,return_counts=True)
    mainHashtags = sorted((zip(mainHashtags[1], mainHashtags[0])), reverse=True)
    sortedNumberHashtags, sortedMainHashtags = zip(*mainHashtags)
    hashtagsOnce = [t[1] for t in mainHashtags if t[0] == 1]
    hashtagsFinales = [hashtag for hashtag in list if hashtag[0] not in hashtagsOnce]
    hashtagsFinales = [hashtag for hashtag in hashtagsFinales if hashtag[1] not in hashtagsOnce]
    return hashtagsFinales