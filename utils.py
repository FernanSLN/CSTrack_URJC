import pandas as pd
import re
import numpy as np


def get_subgraphs(graph):
    import networkx as nx
    components = list(nx.connected_components(graph))
    list_subgraphs = []
    for component in components:
        list_subgraphs.append(graph.subgraph(component))

    return list_subgraphs


def filter_by_topic(df, keywords, stopwords):
    if keywords:
        df = df[df['Texto'].str.contains("|".join(keywords), case=False).any(level=0)]
        if stopwords:
            df = df[~df['Texto'].str.contains("|".join(stopwords), case=False).any(level=0)]
        df.to_csv("learning.csv")
    return df

def get_cites(filename, keywords=None, stopwords=None):
    df = pd.read_csv(filename, sep=';', error_bad_lines=False)
    df = df.drop([78202], axis=0)
    df = filter_by_topic(df, keywords, stopwords)
    dfMentions = df[['Usuario', 'Texto']].copy()
    dfMentions = dfMentions.dropna()
    dfEliminarRTs = dfMentions[dfMentions['Texto'].str.match('RT @')]
    dfMentions = dfMentions.drop(dfEliminarRTs.index)
    mentionsSubset = dfMentions[['Usuario', 'Texto']]
    mentionsList = [list(x) for x in mentionsSubset.to_numpy()]
    return mentionsList





def get_retweets(filename, keywords=None, stopwords=None):
    df = pd.read_csv(filename, sep=';', error_bad_lines=False)
    #stopwords = ['#citizenscience', 'citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana']
    # CALCULAR GRAFO RTs
    df = filter_by_topic(df, keywords, stopwords)
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

def get_hashtagsRT(filename, keywords=None, stopwords=None):
    df = pd.read_csv(filename, sep=';', error_bad_lines=False)
    df = filter_by_topic(df, keywords, stopwords)
    stop_words = ['#citizenscience', 'citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana']
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
    stop_words = ['#citizenscience', 'citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana']
    list = [x.lower() for x in list]
    list = [word for word in list if word not in stop_words]
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

# Código para calcular grafo de hashtags main

def get_hashtagsmain(filename, keywords=None, stopwords=None):
    df = pd.read_csv(filename, sep=';', error_bad_lines=False)
    df = df.drop([78202], axis=0)
    df = filter_by_topic(df, keywords, stopwords)
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
