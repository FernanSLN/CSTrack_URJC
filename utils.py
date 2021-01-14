import pandas as pd
import re
import numpy as  np


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
    dfMentions = df[['Usuario', 'Texto']].copy()
    dfMentions = dfMentions.dropna()
    dfEliminarRTs = dfMentions[dfMentions['Texto'].str.match('RT @')]
    dfMentions = dfMentions.drop(dfEliminarRTs.index)
    mentionsSubset = dfMentions[['Usuario', 'Texto']]
    mentionsList = [list(x) for x in mentionsSubset.to_numpy()]
    return mentionsList


def get_retweets(filename):
    df = pd.read_csv(filename, sep=';', error_bad_lines=False)
    stopwords = ['#citizenscience', 'citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana']
    # CALCULAR GRAFO RTs
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


# Código para hacer gráfica de Hashtags en retuits

def get_hashtagsRT(filename):
    df = pd.read_csv(filename, sep=';', error_bad_lines=False)
    stopwords = ['#citizenscience', 'citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana']
    dfHashtagsRT = df[['Usuario', 'Texto']].copy()
    dfHashtagsRT = dfHashtagsRT.drop([78202], axis=0)
    dfHashtagsRT = dfHashtagsRT.dropna()
    dfHashtagsRT = dfHashtagsRT[dfHashtagsRT['Texto'].str.match('RT:')]
    listHashtagsRT = dfHashtagsRT['Texto'].to_numpy()
    return listHashtagsRT


def get_edgesHashRT(values):
    edges = []
    for row in values:
        match = re.findall('#(\w+)', row)
        for hashtag in match:
            edges.append(hashtag)
    return edges


def prepare_hashtags(list):
    stopwords = ['#citizenscience', 'citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana']
    list = [x.lower() for x in list]
    list = [word for word in list if word not in stopwords]
    list = np.unique(list, return_counts=True)
    list = sorted((zip(list[1], list[0])), reverse=True)
    sortedNumberHashtags, sortedHashtagsRT = zip(*list)
    return sortedNumberHashtags, sortedHashtagsRT


# Código para calcular el grafo de Hashtags dentro de los retuits

def get_hashtagsRT2(filename):
    df = pd.read_csv(filename, sep=';', error_bad_lines=False)
    df = df.drop([78202], axis=0)
    stopwords = ['#citizenscience', 'citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana']
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

