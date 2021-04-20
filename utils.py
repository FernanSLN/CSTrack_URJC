import pandas as pd
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from os import path
from PIL import Image
from collections import Counter
import string
import nltk
from nltk.corpus import stopwords
import matplotlib.dates as mdates
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# stop_words para emplear en filtrados:

stop_words = ['#citizenscience', 'citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana']

# Función para grafica de barras:

def plotbarchart(numberbars, x, y, title, xlabel, ylabel):
    sns.set()
    plt.figure(figsize=(10, 8))
    plt.bar(x=x[:numberbars], height=y[:numberbars], color='lightsteelblue')
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.xticks(rotation=45)
    plt.title(title, fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Función para obtener los subgrafos con NetworkX:

def get_subgraphs(graph):
    import networkx as nx
    components = list(nx.connected_components(graph))
    list_subgraphs = []
    for component in components:
        list_subgraphs.append(graph.subgraph(component))

    return list_subgraphs

# Función para convertir a direct graph los subgrafos:

def direct_subgraphs(subgraphs):
    list_directsubgraphs = []
    for i in range(len(subgraphs)):
        list_directsubgraphs.append(subgraphs[i].to_directed())

    return list_directsubgraphs

# Función para filtrar usando el topic que nos interese:

def filter_by_topic(df, keywords, stopwords):
    if keywords:
        df = df[df['Texto'].str.contains("|".join(keywords), case=False).any(level=0)]
        if stopwords:
            df = df[~df['Texto'].str.contains("|".join(stopwords), case=False).any(level=0)]
    return df

# Función para filtrado por subtopics:

def filter_by_subtopic(df, keywords2, stopwords2):
    if keywords2:
        df = df[df['Texto'].str.contains("|".join(keywords), case=False).any(level=0)]
        if stopwords2:
            df = df[~df['Texto'].str.contains("|".join(stopwords2), case=False).any(level=0)]
    return df

# Función para filtrar por interés:

def filter_by_interest(df, interest):
    if interest is str:
        df = df[df['Marca'] == interest]
    if interest is list:
        df = df[df['Marca'].isin(interest)]
    if interest is None:
        pass
    return df

# Calcular grafo de citas:

def get_cites(filename, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None):
    df = pd.read_csv(filename, sep=';', encoding='latin-1', error_bad_lines=False)
    df = df.drop([78202], axis=0)
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    dfMentions = df[['Usuario', 'Texto']].copy()
    dfMentions = dfMentions.dropna()
    dfEliminarRTs = dfMentions[dfMentions['Texto'].str.match('RT @')]
    dfMentions = dfMentions.drop(dfEliminarRTs.index)
    mentionsSubset = dfMentions[['Usuario', 'Texto']]
    mentionsList = [list(x) for x in mentionsSubset.to_numpy()]
    return mentionsList



# Calcular grafos de RT:

def get_retweets(filename, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None):
    df = pd.read_csv(filename, sep=';', encoding='latin-1', error_bad_lines=False)
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    dfRT = df[['Usuario', 'Texto', 'Fecha']].copy()  # Se copia a un dataframe de trabajo
    idx = dfRT['Texto'].str.contains('RT @', na=False)
    dfRT = dfRT[idx]  # Se seleccionan sólo las filas con RT
    subset = dfRT[['Usuario', 'Texto']]  # Se descarta la fecha
    retweetEdges = [list(x) for x in subset.to_numpy()]  # Se transforma en una lista
    return retweetEdges

# Función para extraer edges de rts y citas:

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


## Código para crear gráfica de barras  de Hashtags más usados en los retuits:
# Seleccionamos las filas solo con RTs y creamos al final una lista que contiene todos los textos

def get_hashtagsRT(filename, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None):
    df = pd.read_csv(filename, sep=';', encoding='latin-1', error_bad_lines=False)
    df = df.drop([78202], axis=0)
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    dfHashtagsRT = df[['Usuario', 'Texto']].copy()
    dfHashtagsRT = dfHashtagsRT.dropna()
    dfHashtagsRT = dfHashtagsRT[dfHashtagsRT['Texto'].str.match('RT @')]
    listHashtagsRT = dfHashtagsRT['Texto'].to_numpy()
    return listHashtagsRT

# Obtenemos los hashtags usados en esos textos

def get_edgesHashRT(values):
    edges = []
    for row in values:
        match = re.findall('#(\w+)', row)
        for hashtag in match:
            edges.append(hashtag)
    return edges

# Organizamos los hashtags en orden de más usados a menos usados y creamos una lista con la cantidad de veces que aparecen
def prepare_hashtags(list):
    stop_words = ['#citizenscience', 'citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana']
    list = [x.lower() for x in list]
    list = [word for word in list if word not in stop_words]
    list = np.unique(list, return_counts=True)
    list = sorted((zip(list[1], list[0])), reverse=True)
    sortedNumberHashtags, sortedHashtagsRT = zip(*list)
    return sortedNumberHashtags, sortedHashtagsRT


# Código para calcular el grafo de Hashtags dentro de los retuits

def get_hashtagsRT2(filename, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None):
    df = pd.read_csv(filename, sep=';', encoding='latin-1', error_bad_lines=False)
    df = df.drop([78202], axis=0)
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
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

# Combinación de los ejes de RTs y Citas:

def combined_edges(x,y):
    combined_edges = x + y
    return combined_edges

## Código para calcular grafo de hashtags relacionados fuera de RTs. Este grafo opretende mostrar que
##hashtags estan interrlacionados entre si.

def get_hashtagsmain(filename, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None):
    df = pd.read_csv(filename, sep=';', encoding='latin-1', error_bad_lines=False)
    df = df.drop([78202], axis=0)
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    dfMainHashtags = df[['Usuario', 'Texto']].copy()
    dfMainHashtags = dfMainHashtags.dropna()
    idx = dfMainHashtags[dfMainHashtags['Texto'].str.match('RT @')]
    dfMainHashtags = dfMainHashtags.drop(idx.index)
    subset = dfMainHashtags['Texto']
    listMainHashtags = subset.to_numpy()
    return listMainHashtags

def mainHashtags(values):
    stop_words = ['#citizenscience', 'citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana']
    mainHashtags = []
    aristasHashtags = []
    for row in values:
        match = re.findall('#(\w+)', row.lower())
        length = len(match)
    try:
        match = [word for word in match if word not in stop_words]
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

# Creación de grafo hashtags más utilizados (relacionado con usuario):

def get_hashtagsmain2(filename, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None):
    df = pd.read_csv(filename, sep=';', encoding='latin-1', error_bad_lines=False)
    df = df.drop([78202], axis=0)
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    dfMainHashtags = df[['Usuario', 'Texto']].copy()
    dfMainHashtags = dfMainHashtags.dropna()
    idx = dfMainHashtags[dfMainHashtags['Texto'].str.match('RT @')]
    dfMainHashtags = dfMainHashtags.drop(idx.index)
    subset = dfMainHashtags[['Usuario','Texto']]
    listMainHashtags = [list(x) for x in subset.to_numpy()]
    return listMainHashtags

def get_edgesmain2(values):
    stop_words = [['citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana','CitizenScience']]
    edges = []
    for row in values:
        match = re.search('#(\w+)', row[1])
        if match:
            matchhash = match.group(1)
            row[1] = matchhash
            edges.append(row)
            edges = [i for i in edges if i[1] != stop_words]
    return edges

# Creación de gráfica hashtags más usados fuera de RTs (usar get_hashtagsmain())

def get_edgesMain(values):
    edges = []
    for row in values:
        match = re.findall('#(\w+)', row.lower())
        for hashtag in match:
            edges.append(hashtag)
    return edges

# Hashtags del Bot:
botwords=['airpollution', 'luftdaten', 'fijnstof', 'waalre', 'pm2', 'pm10']

def prepare_hashtagsmain(list, stopwords=None):
    citsci_words = ['#citizenscience', 'citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana']
    lista = [x.lower() for x in list]
    lista = [word for word in lista if word not in citsci_words]
    lista = [word for word in lista if word not in stopwords]
    mainHashtags = np.unique(lista,return_counts=True)
    mainHashtags = sorted((zip(mainHashtags[1], mainHashtags[0])), reverse=True)
    sortedNumberHashtags, sortedMainHashtags = zip(*mainHashtags)
    return sortedNumberHashtags,sortedMainHashtags


def get_prop_type(value, key=None):
    """
    Performs typing and value conversion for the graph_tool PropertyMap class.
    If a key is provided, it also ensures the key is in a format that can be
    used with the PropertyMap. Returns a tuple, (type name, value, key)
    """

    # Deal with the value
    if isinstance(value, bool):
        tname = 'bool'

    elif isinstance(value, int):
        tname = 'float'
        value = float(value)

    elif isinstance(value, float):
        tname = 'float'

    elif isinstance(value, dict):
        tname = 'object'

    else:
        tname = 'string'
        value = str(value)

    return tname, value, key


def nx2gt(nxG):
    """
    Converts a networkx graph to a graph-tool graph.
    """
    # Phase 0: Create a directed or undirected graph-tool Graph
    gtG = gt.Graph(directed=nxG.is_directed())

    # Add the Graph properties as "internal properties"
    for key, value in nxG.graph.items():
        # Convert the value and key into a type for graph-tool
        tname, value, key = get_prop_type(value, key)

        prop = gtG.new_graph_property(tname)  # Create the PropertyMap
        gtG.graph_properties[key] = prop      # Set the PropertyMap
        gtG.graph_properties[key] = value     # Set the actual value

    # Phase 1: Add the vertex and edge property maps
    # Go through all nodes and edges and add seen properties
    # Add the node properties first
    nprops = set()  # cache keys to only add properties once
    for node, data in nxG.nodes(data=True):

        # Go through all the properties if not seen and add them.
        for key, val in data.items():
            if key in nprops:
                continue  # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key = get_prop_type(val, key)

            prop = gtG.new_vertex_property(tname)  # Create the PropertyMap
            gtG.vertex_properties[key] = prop      # Set the PropertyMap

            # Add the key to the already seen properties
            nprops.add(key)

    # Also add the node id: in NetworkX a node can be any hashable type, but
    # in graph-tool node are defined as indices. So we capture any strings
    # in a special PropertyMap called 'id' -- modify as needed!
    gtG.vertex_properties['id'] = gtG.new_vertex_property('string')

    # Add the edge properties second
    eprops = set()  # cache keys to only add properties once
    for src, dst, data in nxG.edges(data=True):

        # Go through all the edge properties if not seen and add them.
        for key, val in data.items():
            if key in eprops:
                continue  # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key = get_prop_type(val, key)

            prop = gtG.new_edge_property(tname)  # Create the PropertyMap
            gtG.edge_properties[key] = prop      # Set the PropertyMap

            # Add the key to the already seen properties
            eprops.add(key)

    # Phase 2: Actually add all the nodes and vertices with their properties
    # Add the nodes
    vertices = {}  # vertex mapping for tracking edges later
    for node, data in nxG.nodes(data=True):

        # Create the vertex and annotate for our edges later
        v = gtG.add_vertex()
        vertices[node] = v

        # Set the vertex properties, not forgetting the id property
        data['id'] = str(node)
        for key, value in data.items():
            gtG.vp[key][v] = value  # vp is short for vertex_properties

    # Add the edges
    for src, dst, data in nxG.edges(data=True):

        # Look up the vertex structs from our vertices mapping and add edge.
        e = gtG.add_edge(vertices[src], vertices[dst])

        # Add the edge properties
        for key, value in data.items():
            gtG.ep[key][e] = value  # ep is short for edge_properties

    # Done, finally!
    return gtG

# Función para extraer los valores de degree,outdegree, eigenvector y betweenness y crear un csv:

def csv_degval(Digraph, filename):
    list_values = []
    outdegrees2 = dict(Digraph.out_degree())
    indegrees = dict(Digraph.in_degree())
    centrality = dict(nx.eigenvector_centrality(Digraph))
    betweenness = dict(nx.betweenness_centrality(Digraph))
    indegtupl = sorted([(k, v) for k, v in indegrees.items()], key=lambda x:x[1], reverse=True)
    indegtupl = indegtupl[0:10]
    names = [i[0] for i in indegtupl]
    outdegtupl = sorted([(k,v) for k,v in outdegrees2.items()], key=lambda x:x[1], reverse=True)
    centraltupl = sorted([(k,v) for k,v in centrality.items()], key=lambda x:x[1], reverse=True)
    betwentupl = sorted([(k,v) for k,v in betweenness.items()], key=lambda x:x[1], reverse=True)
    for name in names:
        pos_indeg = [y[0] for y in indegtupl].index(name)
        rank_indeg = pos_indeg + 1
        indeg_val = indegtupl[pos_indeg][1]
        pos_outdeg = [y[0] for y in outdegtupl].index(name)
        rank_outdeg = pos_outdeg + 1
        outdeg_val = outdegtupl[pos_outdeg][1]
        pos_central = [y[0] for y in centraltupl].index(name)
        rank_central = pos_central + 1
        central_val = centraltupl[pos_central][1]
        pos_between = [y[0] for y in betwentupl].index(name)
        rank_between = pos_between + 1
        between_val = betwentupl[pos_between][1]
        list_values.append((name, indeg_val, rank_indeg, outdeg_val, rank_outdeg, central_val, rank_central,
                        between_val, rank_between))
    df = pd.DataFrame(list_values,
                      columns=['Name', 'Indegree', 'Rank', 'Outdegree', 'Rank', 'Eigenvector', 'Rank', 'Betweenness',
                               'Rank'])
    return df.to_csv(filename, index=False)

## Funciones para obtener los elementos de la two mode:
# Obtención de los elementos u,v y los edges que los unen para usuario y texto en los retuits:

def get_uv_edgesRT(filename, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None):
    edges = []
    df = pd.read_csv(filename, sep=';', error_bad_lines=False)
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    dfRT = df[['Usuario', 'Texto']].copy()
    idx = dfRT['Texto'].str.contains('RT @', na=False)
    dfRT = dfRT[idx]
    subset = dfRT[['Usuario', 'Texto']]
    u = list(subset['Usuario'])
    v = list(subset['Texto'])
    edges = [tuple(x) for x in subset.to_numpy()]
    return edges, u, v

# Obtención de los elemetnos u,v y los edges para los hashtags fuera de los retuits:

def get_uv_HashMain(filename, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None, filter_hashtags=None):
    edges = []
    df = pd.read_csv(filename, sep=';', error_bad_lines=False, encoding='utf-8')
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    dfMain = df[['Usuario', 'Texto']].copy()
    dfMain = dfMain.dropna()
    dfEliminarRTs = dfMain[dfMain['Texto'].str.match('RT @')]
    dfMain = dfMain.drop(dfEliminarRTs.index)
    subset = dfMain[['Usuario', 'Texto']]
    listHT = [list(x) for x in subset.to_numpy()]
    stop_words = [['citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana','CitizenScience']]
    for row in listHT:
        match = re.search('#(\w+)', row[1])
        if match:
            matchhash = match.group(1)
            row[1] = matchhash
            edges.append(row)
    if filter_hashtags == True:
        filter_edges = []
        for edge in edges:
            stop = False
            for word in edge:
                # print(word, word.lower() in stop_words)
                if word.lower() in stop_words:
                    stop = True
            if not stop:
                filter_edges.append(edge)
    u = [x[0] for x in filter_edges]
    v = [x[1] for x in filter_edges]
    return filter_edges, u, v

# Función para obtener los componentes de la two mode para hashtags en retuits:

def getuv_htRT(filename, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None, filter_hashtags=None):
    edges = []
    stop_words = ['CitizenScience', 'citizenScience','citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana', '#CitizenScience']
    df = pd.read_csv(filename, sep=';', error_bad_lines=False, encoding='utf-8')
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    df = df[['Usuario', 'Texto']].copy()
    df = df.dropna()
    idx = df['Texto'].str.contains('RT @', na=False)
    dfRT = df[idx]  # Se seleccionan sólo las filas conpython  RT
    subset = dfRT[['Usuario', 'Texto']]
    listHT = [list(x) for x in subset.to_numpy()]
    for row in listHT:
        match = re.search('#(\w+)', row[1])
        if match:
            matchhash = match.group(1)
            row[1] = matchhash
            edges.append(row)
    if filter_hashtags == True:
        filter_edges = []
        for edge in edges:
           stop = False
           for word in edge:
               #print(word, word.lower() in stop_words)
                if word.lower() in stop_words:
                    stop = True
           if not stop:
               filter_edges.append(edge)
    u = [x[0] for x in filter_edges]
    v = [x[1] for x in filter_edges]
    return filter_edges, u, v


# Wordcloud function for main hashtags:

def wordcloudmain(filename, keywords=None, stopwords=None, interest=None ):
    hashtags =[]
    stop_words = ['citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana', 'CitizenScience']
    df = pd.read_csv(filename, sep=';', encoding='latin-1', error_bad_lines=False)
    df = df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = df[['Usuario', 'Texto']]
    df = df.dropna()
    idx = df[df['Texto'].str.match('RT @')]
    df = df.drop(idx.index)
    subset = df['Texto']
    for row in subset:
        match = re.findall('#(\w+)', row.lower())
        for hashtag in match:
            hashtags.append(hashtag)
    unique_string = (' ').join(hashtags)
    wordcloud = WordCloud(width=900, height=900, background_color='white', stopwords=stop_words,
                          min_font_size=10, max_words=10405, collocations=False, colormap='winter').generate(unique_string)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

# Wordcloud for main hashtags plotted inside a logo:

def transform_format(val):
    if val == 0:
        return 255
    else:
        return val


def wordcloud_mainhtlogo(filename, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None, image=None):
    hashtags =[]
    stop_words = ['citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana', 'CitizenScience']
    df = pd.read_csv(filename, sep=';', encoding='latin-1', error_bad_lines=False)
    df = df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    df = df[['Usuario', 'Texto']]
    df = df.dropna()
    idx = df[df['Texto'].str.match('RT @')]
    df = df.drop(idx.index)
    subset = df['Texto']
    for row in subset:
        match = re.findall('#(\w+)', row.lower())
        for hashtag in match:
            hashtags.append(hashtag)
    unique_string = (' ').join(hashtags)
    logo = np.array(Image.open(image))
    transformed_logo = np.ndarray((logo.shape[0], logo.shape[1]), np.int32)

    for i in range(len(logo)):
        transformed_logo[i] = list(map(transform_format, logo[i]))

    wc = WordCloud(width = 900, height = 900,
                background_color ='ghostwhite',
                stopwords = stop_words,
                min_font_size = 5, max_font_size=30, max_words=10405, collocations=False,mask=transformed_logo,
          contour_width=2, contour_color='cornflowerblue',mode='RGB', colormap='summer').generate(unique_string)

    plt.figure(figsize=[25, 10])
    plt.imshow(wc)
    plt.axis("off")
    plt.show()


# Wordlcoud for hashtags in the RTs:

def wordcloudRT(filename, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None ):
    hashtags =[]
    stop_words = ['citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana', 'CitizenScience']
    df = pd.read_csv(filename, sep=';', encoding='latin-1', error_bad_lines=False)
    df = df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    df = df[['Usuario', 'Texto']]
    df = df.dropna()
    idx = df['Texto'].str.contains('RT @', na=False)
    df = df[idx]
    subset = df['Texto']
    for row in subset:
        match = re.findall('#(\w+)', row.lower())
        for hashtag in match:
            hashtags.append(hashtag)
    unique_string = (' ').join(hashtags)
    wordcloud = WordCloud(width=900, height=900, background_color='white', stopwords=stop_words,
                          min_font_size=10, max_words=10405, collocations=False, colormap='winter').generate(unique_string)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

def wordcloudRT_logo(filename, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None, image=None):
    hashtags = []
    stop_words = ['citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana', 'CitizenScience']
    df = pd.read_csv(filename, sep=';', encoding='latin-1', error_bad_lines=False)
    df = df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    df = df[['Usuario', 'Texto']]
    df = df.dropna()
    idx = df['Texto'].str.contains('RT @', na=False)
    df = df[idx]
    subset = df['Texto']
    for row in subset:
        match = re.findall('#(\w+)', row.lower())
        for hashtag in match:
            hashtags.append(hashtag)
    unique_string = (' ').join(hashtags)

    logo = np.array(Image.open(image))
    transformed_logo = np.ndarray((logo.shape[0], logo.shape[1]), np.int32)

    for i in range(len(logo)):
        transformed_logo[i] = list(map(transform_format, logo[i]))

    wc = WordCloud(width=900, height=900,
                   background_color='ghostwhite',
                   stopwords=stop_words,
                   min_font_size=5, max_font_size=30, max_words=10405, collocations=False, mask=transformed_logo,
                   contour_width=2, contour_color='cornflowerblue', mode='RGB', colormap='summer').generate(
        unique_string)

    plt.figure(figsize=[25, 10])
    plt.imshow(wc)
    plt.axis("off")
    plt.show()


# Cálculo de las palabras más usadas:
# La función emplea la columna texto y podemos añadir un número n que indica cuantas palabras

def most_common(filename,number=None):
    df = pd.read_csv(filename, sep=';', encoding='latin-1', error_bad_lines=False)
    subset = df['Texto']
    subset = subset.dropna()
    # Definimos stopwords en varios idiomas y símbolos que queremos eliminar del resultado
    s = stopwords.words('english')
    e = stopwords.words('spanish')
    r = STOPWORDS
    d = stopwords.words('german')
    p = string.punctuation
    new_elements = ('\\n', 'rt', '?', '¿', '&', 'that?s', '??', '-', '???')
    s.extend(new_elements)
    s.extend(e)
    s.extend(r)
    s.extend(d)
    s.extend(p)
    s = set(s)
    # Calculamos la frecuencia de las palabras
    word_freq = Counter(" ".join(subset).lower().split())
    for word in s:
        del word_freq[word]
    return word_freq.most_common(number)

# Top palabras más usadas en wordcloud:

def most_commonwc(filename):
    df = pd.read_csv(filename, sep=';', encoding='latin-1', error_bad_lines=False)
    subset = df['Texto']
    subset = subset.dropna()
    s = stopwords.words('english')
    e = stopwords.words('spanish')
    r = STOPWORDS
    d = stopwords.words('german')
    p = string.punctuation
    new_elements = ('\\n', 'rt', '?', '¿', '&', 'that?s', '??', '-','the', 'to')
    s.extend(new_elements)
    s.extend(e)
    s.extend(r)
    s.extend(d)
    s.extend(p)
    stopset = set(s)
    word_freq = Counter(" ".join(subset).lower().split())
    for word in s:
        del word_freq[word]
    wordcloud = WordCloud(width=900, height=900, background_color='white', stopwords=stopset,
                          min_font_size=10, max_words=10405, collocations=False,
                          colormap='winter').generate_from_frequencies(word_freq)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

# Gráficos temporales

def main_or_RT_days(filename, RT=None):
    df = pd.read_csv(filename, sep=';', encoding='utf-8', error_bad_lines=False)
    df = df[['Fecha', 'Usuario', 'Texto']]
    df = df.dropna()
    if RT == True:
        idx = df['Texto'].str.contains('RT @', na=False)
        subset = df[idx]
    else:
        dfEliminarRTs = df[df['Texto'].str.match('RT @')]
        subset = df.drop(dfEliminarRTs.index)

    subset['Fecha'] = pd.to_datetime(subset['Fecha'], errors='coerce')
    subset = subset.dropna()
    subset['Fecha'] = subset['Fecha'].dt.date

    # Obtenemos los días en el subset:
    df_Fecha = subset['Fecha']
    days = pd.unique(df_Fecha)
    days.sort()

    return subset, days

# Función para graficar uso de hashtags en el tiempo. En df utilizar Maindf o dfRT obtenidos con main_or_RT_days.
# Days emplear los obtenidos en la función anterior también. Elements la lista de hashtags
# ordenados sortedMH (main hashtags) o sortedHT (RT) obtenidos con las funciones listHT/listHRT- get_edgesMain/
# get_EdgesHashRT- preparehashtagsmain/preparehashtags:

def plottemporalserie(days, df, elements, title, x=None, y=None):
    numHashtag = []
    for hashtag in elements[x:y]:
        numPerDay = []
        for day in days:
            dfOneDay = df[df['Fecha'] == day]
            count = dfOneDay['Texto'].str.contains(hashtag, case=False).sum()
            numPerDay.append(count)
        numHashtag.append(numPerDay)

    sns.reset_orig()
    fig = plt.figure(figsize=(9, 6))

    i = 0
    for hashtag in elements[x:y]:
        plt.plot_date(days, numHashtag[i], linestyle='solid', markersize=0, label=hashtag)
        i += 1

        # Se fija el titulo y etiquetas
    plt.title(title, fontsize=20, fontweight='bold')
    plt.xlabel("Fecha", fontsize=15)
    plt.ylabel("Número de veces", fontsize=15)
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    fig.autofmt_xdate()
    plt.show()

# plot temporal series de un hashtag a nuestra elección (variable name), el resto igual:

def one_hastag_temporalseries(df, elements, days, name, title):
    numHashtag = []
    for i in elements:
        if i == name:
            for day in days:
                numPerDay = []
                dfOneDay = df[df['Fecha'] == day]
                count = dfOneDay['Texto'].str.contains(i, case=False).sum()
                numPerDay.append(count)
                numHashtag.append(numPerDay)
            sns.reset_orig()
            fig = plt.figure(figsize=(9, 6))

            plt.plot_date(days, numHashtag, linestyle='solid', color='mediumseagreen', markersize=0, label=name)

            # Se fija el titulo y etiquetas
            plt.title(title, fontsize=20, fontweight='bold')
            plt.xlabel("Fecha", fontsize=15)
            plt.ylabel("Número de veces", fontsize=15)
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

            fig.autofmt_xdate()
            plt.show()

    else:
        print(name + 'not in list')
start_time = time.time()

# Función para obtener los 50 tweets con mayor Impacto/Opinión y los usuarios con mayor impacto/opinión:

def impact_opinionRT(filename, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None, Impact=None, Opinion=None, n=None):
    df = pd.read_csv(filename, sep=';', encoding='utf-8', error_bad_lines=False, decimal=',', dtype={'Impacto':'float64'})
    df = df[['Texto', 'Usuario', 'Opinion', 'Impacto']]
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    df = df.dropna()
    df['Opinion'] = df['Opinion'].replace(',', '.', regex=True).astype(float)
    idx = df['Texto'].str.contains('RT @', na=False)
    df = df[idx]
    df = df.dropna()
    if Impact == True:
        w = 'Impact'
        df = df[['Texto', 'Impacto']]
        df = df.sort_values('Impacto', ascending=False)
        df = df.drop_duplicates(subset='Texto', keep='first')
        subset = df[['Texto', 'Impacto']]
    else:
        if Opinion == True:
            w = 'Opinion'
            df = df[['Texto', 'Opinion']]
            df = df.sort_values('Opinion', ascending=False)
            df = df.drop_duplicates(subset='Texto', keep='first')
            subset = df[['Texto', 'Opinion']]
    retweets = [list(x) for x in subset.to_numpy()]
    CSV = subset[:50].to_csv('top50 retweets by ' + w + '.csv', sep=';', index=False, decimal='.', encoding='utf-8')
    arrobas = []
    for row in retweets:
        reg = re.search('@(\w+)', row[0])
        if reg:
            matchRT = reg.group(1)
            row[0] = matchRT
            arrobas.append(row)
    df_arrobas = pd.DataFrame(arrobas, columns=['User', 'Values'])
    df_arrobas = df_arrobas.groupby('User', as_index=False).mean()
    df_arrobas = df_arrobas.sort_values('Values', ascending=False)
    users = df_arrobas['User']
    values = df_arrobas['Values']
    plotbarchart(n, users, values, 'Top ' + str(n) + ' Users with higher ' + w, 'User', w)


def impact_opinion(filename, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None, Impact=None, Opinion=None, n=None):
    df = pd.read_csv(filename, sep=';', encoding='utf-8', error_bad_lines=False, decimal=',',
                     dtype={'Impacto': 'float64'})
    df = df[['Texto', 'Usuario', 'Opinion', 'Impacto']]
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    df = df.dropna()
    df['Opinion'] = df['Opinion'].replace(',', '.', regex=True).astype(float)
    idx = df[df['Texto'].str.contains('RT @', na=False)]
    df = df.drop(idx.index)
    df = df.dropna()
    if Impact == True:
        w = 'Impact'
        df = df[['Usuario', 'Texto', 'Impacto']]
        df = df.sort_values('Impacto', ascending=False)
        df = df.drop_duplicates(subset='Texto', keep='first')
        subset = df[['Usuario', 'Texto', 'Impacto']]
    else:
        if Opinion == True:
            w = 'Opinion'
            df = df[['Usuario', 'Texto', 'Opinion']]
            df = df.sort_values('Opinion', ascending=False)
            df = df.drop_duplicates(subset='Texto', keep='first')
            subset = df[['Usuario', 'Texto', 'Opinion']]
    CSV = subset[:50].to_csv('top50 tweets by ' + w + '.csv', sep=';', index=False, decimal='.', encoding='utf-8')
    lista = [list(x) for x in subset.to_numpy()]
    df_Users = pd.DataFrame(lista, columns=['User', 'Texto', 'Values'])
    df_Users = df_Users[['User', 'Values']]
    df_Users = df_Users.groupby('User', as_index=False).mean()
    df_Users = df_Users.sort_values('Values', ascending=False)
    users = df_Users['User']
    values = df_Users['Values']
    plotbarchart(n, users, values, 'Top ' + str(n) + ' Users with higher ' + w, 'User', w)

# Analisis de sentimiento con VaderSentiment

analyser = SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(sentence):
        score = analyser.polarity_scores(sentence)
        return score

def sentiment_analyser(filename,keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None):
    df = pd.read_csv(filename, sep=';', encoding='utf-8', error_bad_lines=False)
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    df = df[['Texto', 'Usuario']]
    df = df.dropna()
    Users = df['Usuario']
    Texto = df['Texto']
    sentences = Texto
    list_of_dicts = []
    for sentence in sentences:
        adict = analyser.polarity_scores(sentence)
        list_of_dicts.append(adict)
    df_sentiment = pd.DataFrame(list_of_dicts)
    df_sentiment['Usuario'] = Users
    df_sentiment['Texto'] = Texto
    df_sentiment = df_sentiment[['Usuario', 'Texto', 'compound']]
    df_sentiment['compound'] = df_sentiment.compound.multiply(100)
    df_sentiment.rename({'compound':'Sentiment'}, axis='columns')
    CSV = df_sentiment.to_csv('vaderSentiment.csv', sep=';', decimal='.', encoding='utf-8')
    return CSV

# Gráfico de barras y csv de users top en Sentiment:

def sentiment_resultsRT(filename, n=None):
    df_sentiment = pd.read_csv(filename, sep=';', encoding='utf-8', error_bad_lines=False)
    df_sentiment = df_sentiment[['Texto', 'compound']]
    idx = df_sentiment['Texto'].str.contains('RT @', na=False)
    df_sentimentRT = df_sentiment[idx]
    retweets = [list(x) for x in df_sentimentRT.to_numpy()]
    arrobas = []
    for row in retweets:
        reg = re.search('@(\w+)', row[0])
        if reg:
            matchRT = reg.group(1)
            row[0] = matchRT
            arrobas.append(row)

    df_arrobas = pd.DataFrame(arrobas, columns=['User', 'Values'])
    df_arrobas = df_arrobas.groupby('User', as_index=False).mean()
    df_arrobas = df_arrobas.sort_values('Values', ascending=False)
    users = df_arrobas['User']
    values = df_arrobas['Values']
    plotbarchart(n, users, values, 'Top' + str(n) + 'Retweeted Users with higher Sentiment', 'User', 'Sentiment')
    df_SRT = df_sentimentRT.sort_values('compound', ascending=False)
    dfSRT = df_SRT.drop_duplicates(subset='Texto', keep='first')
    df_SRT[:50].to_csv('top50 retweets by Sentiment.csv', sep=';', index=False, decimal='.', encoding='utf-8')
    return df_arrobas

def sentiment_results(filename, n=None):
    df_sentiment = pd.read_csv(filename, sep=';', encoding='utf-8', error_bad_lines=False)
    idx = df_sentiment[df_sentiment['Texto'].str.contains('RT @', na=False)]
    df_sentiment = df_sentiment.drop(idx.index)
    df_sentiment = df_sentiment.sort_values('compound', ascending=False)
    df_sentiment = df_sentiment.drop_duplicates(subset='Texto', keep='first')
    df_sentiment[:50].to_csv('top 50 tweets by Sentiment.csv', sep=';', index=False, decimal='.', encoding='utf-8')
    subset = df_sentiment[['Usuario', 'compound']]
    subset = subset.groupby('Usuario', as_index=False).mean()
    subset = subset.sort_values('compound', ascending=False)
    subset = subset.rename(columns={'Usuario': 'User', 'compound': 'Values'})
    users = subset['User']
    values = subset['Values']
    plotbarchart(n, users, values, 'Top' + str(n) + 'Users with higher Sentiment', 'User', 'Sentiment')
    return subset

# Combinación de ambos subsets y cálculo absoluto de los usuarios con mayor Sentiment calculado en Vader:

def combined_vader(subset1, subset2, n=None):
    frames = [subset1, subset2]
    vader_df = pd.concat(frames, axis=0)
    vader_df = vader_df.groupby('User', as_index=False).mean()
    vader_df = vader_df.sort_values('Values', ascending=False)
    users = vader_df['User']
    values = vader_df['Values']
    plotbarchart(n, users, values, 'Top' + str(n) + 'Users with higher Sentiment', 'User', 'Sentiment')

# Adición de pesos a lista de ejes y creación de DIGraph:
# Primera función añade peso como caracteristica en dict de los ejes:

def make_weightedDIGraph(ejes):
    edges_tupla = [tuple(x) for x in ejes]
    G = nx.DiGraph((x, y, {'weight': v}) for (x, y), v in Counter(edges_tupla).items())
    return G

# Segunda función añade peso como tercer elemento de la tupla (nodo, nodo, peso):

def weighted_DiGraph(ejes):
    for lista in ejes:
        lista.append(1)

    result = Counter()

    for k, v, z in ejes:
        result.update({(k,v):z})

    result = dict(result)

    tupla_ejes_peso = []

    for key, value in result.items():
        temp = [key, value]
        tupla_ejes_peso.append(temp)

    ejes_tupla = []

    pesos = []
    for lista in tupla_ejes_peso:
        ejes_tupla.append(lista[0])
        pesos.append(lista[1])

    ejes_lista = [list(x) for x in ejes_tupla]

    for num in range(len(pesos)):
        ejes_lista[num].append(pesos[num])

    ejes_pesos = [tuple(x) for x in ejes_lista]

    G = nx.DiGraph()
    G.add_weighted_edges_from(ejes_pesos)
    return G

#DataFrame con el cálculo de la media, mediana y sd de Impacto y Opinión:

def dataframe_statistics(filename, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None):
    df = pd.read_csv(filename, sep=';', encoding='utf-8', error_bad_lines=False,decimal=',', low_memory=False)
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    df = df[['Opinion', 'Impacto']]
    df = df.dropna()
    df['Opinion'] = df['Opinion'].replace(',', '.', regex=True).astype(float)
    mean_opinion = round((df['Opinion'].mean()),2)
    mean_impact = round((df['Impacto'].mean()),2)
    median_opinion = round((df['Opinion'].median()),2)
    median_impact = round((df['Impacto'].median()),2)
    std_opinion = round((df['Opinion'].std()),2)
    std_impact = round((df['Impacto'].std()),2)
    Opinion = [mean_opinion, median_opinion, std_opinion]
    Impact = [mean_impact, median_impact, std_impact]
    d = {'Opinion': Opinion, 'Impact': Impact}
    df_statistics = pd.DataFrame(d, index=['Mean', 'Median', 'Standard deviation'])
    return df_statistics

# Creacion del Data Frame conteniendo los parámetros de análisis estructural de un grafo dirigido:

def graph_structural_analysis(Graph):
    n_nodes = nx.number_of_nodes(Graph)
    n_edges = nx.number_of_edges(Graph)
    density = nx.density(Graph)
    while True:
        try:
            avg_path = nx.average_shortest_path_length(Graph, weight='weight')
            break
        except nx.NetworkXError:
            avg_path = 'not weakly connected, average path length not computable'
            break

    clustering = nx.average_clustering(Graph)
    transitivity = nx.transitivity(Graph)
    while True:
        try:
            diameter = nx.diameter(Graph)
            break
        except nx.NetworkXError:
            diameter = 'not weakly connected, diameter is infinite'
            break

    values = [n_nodes, n_edges, density, avg_path, clustering, transitivity, diameter]
    d = {'Values': values}
    df_structure = pd.DataFrame(data=d, index=['number of nodes', 'number of edges', 'density', 'average path length',
                                               'clustering', 'transitivity', 'diameter'])
    return df_structure

#edge weight distribution:

def edge_weight_distribution(Graph):
    dict = nx.get_edge_attributes(Graph, 'weight')
    peso = dict.values()
    counter = Counter(peso)
    frecuencias = list(counter.values())
    pesos = list(counter.keys())
    numberbars = len(pesos)
    xlabel = 'weights'
    ylabel = 'Frecuency'
    title = 'Edge weight distribution'
    x = pesos
    y = frecuencias
    sns.set()
    plt.figure(figsize=(10, 8))
    plt.bar(x=x[:numberbars], height=y[:numberbars], color='peachpuff', alpha=1, width=1, linewidth=0)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.ticklabel_format(style='sci')
    plt.title(title, fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.show()



