import pandas as pd
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import networkx as nx
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from collections import Counter
import string
import nltk
from nltk.corpus import stopwords
from nltk. tag import pos_tag
from nltk import word_tokenize
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import hashlib

# stop_words to apply filtering:

stop_words = ['#citizenscience', 'citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana']

# Function to create a bargraph:

def plotbarchart(numberbars, x, y, title=None, xlabel=None, ylabel=None):
    """
    Given a number of elements to plot and the elements in x and y axis the function returns a barchart

    :param numberbars: Number of elements to plot in the chart
    :param x: Elements for x axis
    :param y: Elements for y axis, number of appearances of the x elements
    :param title: Title for the figure, defaults to None
    :param xlabel: Label for the x axis, defaults to None
    :param ylabel: Label for the y axis, defaults to None

    """
    sns.set()
    fig, ax = subplots()
    ax.bar(x[:numberbars], y[:numberbars], color="lightsteelblue")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.xticks(rotation=45)
    plt.title(title, fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Scatter plot:

def scatterplot(x, y):
    """
    Given the elements for x axis and the number of the elements for y axis the function returns a scatterplot

    :param x: Elements for the x axis
    :param y: Elements for the y axis, number of appearances of the x elements

    """
    plt.figure(figsize=(10, 8))
    plt.scatter(x=x, y=y, c="lightsteelblue")
    plt.xlabel("Indegree", fontsize=15)
    plt.ylabel("Outdegree", fontsize=15)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
# Function to obtain subgraph in NetworkX:

def get_subgraphs(graph):
    """
    Given a networkx Graph the function returns the subgraphs stored in a list

    :param graph: Networkx undirected graph
    :return: list of subgraphs as networkx objects
    """
    components = list(nx.connected_components(graph))
    list_subgraphs = []
    for component in components:
        list_subgraphs.append(graph.subgraph(component))

    return list_subgraphs

# Converts subgraphs to direct graphs:

def direct_subgraphs(subgraphs):
    """
    Given a networkx undirected list of subgraph the function returns all the graphs as directed

    :param subgraphs: List of undirected networkx subgraphs
    :return: List of directed subgraphs as networkx objects

    """
    list_directsubgraphs = []
    for i in range(len(subgraphs)):
        list_directsubgraphs.append(subgraphs[i].to_directed())

    return list_directsubgraphs

# Funcition to filter by topic:

def filter_by_topic(df, keywords, stopwords):
    """
    Given a DataFrame the function returns the dataframe filtered according the given keywords and stopwords

    :param df: Dataframe with all the tweets
    :param keywords: List of words acting as key to filter the dataframe
    :param stopwords: List of words destined to filter out the tweets containing them
    :return: DataFrame with the tweets containing the keywords
    """
    if keywords:
        df = df[df['Texto'].str.contains("|".join(keywords), case=False).any(level=0)]
        if stopwords:
            df = df[~df['Texto'].str.contains("|".join(stopwords), case=False).any(level=0)]
    return df

# Funcition to filter by subtopic:

def filter_by_subtopic(df, keywords2, stopwords2):
    """
    Given a previously filtered DataFrame the function returns the dataframe filtered according to the new subtopic of interest

    :param keywords2: List of words acting as key to filter the dataframe
    :param stopwords2: List of words destined to filter out the tweets that contain them
    :return: DataFrame with the tweets containing the keywords
    """
    if keywords2:
        df = df[df['Texto'].str.contains("|".join(keywords2), case=False).any(level=0)]
        if stopwords2:
            df = df[~df['Texto'].str.contains("|".join(stopwords2), case=False).any(level=0)]
    return df

# Function to filter by interest:

def filter_by_interest(df, interest):
    """
    Given a non filtered DataFrame the function returns the dataframe filtered by the column interest

    :param df: DataFrame with all the tweets
    :param interest: Active interest from the different categories available from the Lynguo tool
    :return: DataFrame containing the tweets filtered by the selected interest
    """
    if interest is str:
        df = df[df['Marca'] == interest]
    if interest is list:
        df = df[df['Marca'].isin(interest)]
    if interest is None:
        pass
    return df

# Calculate Mentions network graph:

def get_cites(df, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None):
    """
    Given a DataFrame containing tweets the function returns those tweets belonging to the citations type, removing the retweets.
    The function also applies the filtering processes

    :param df: DataFrame with all the tweets
    :param keywords: List of words acting as key to filter the DataFrame
    :param stopwords: List of words destined to filter out the tweets that contain them
    :param keywords2: List of words acting as key to filter the DataFrame according to a subtopic
    :param stopwords2: List of words destined to filter out the tweets that contain them according to a subtopic
    :param interest: Active interest from the different categories available from the Lynguo tool
    :return: Nested lists containing normal tweet, not retweets, and user who wrote the tweet
    """
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



# Calculate RT network graph:

def get_retweets(df, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None):
    """
    Given a DataFrame containing all the tweets the function returns all those tweets which are retweets (RT:@).
    It also applies the filtering procces

    :param df: DataFrame with all the tweets
    :param keywords: List of words acting as key to filter the DataFrame
    :param stopwords: List of words destined to filter out the tweets that contain them
    :param keywords2: List of words acting as key to filter the DataFrame according to a subtopic
    :param stopwords2: List of words destined to filter out the tweets that contain them according to a subtopic
    :param interest: Active interest from the different categories available from the Lynguo tool
    :return: Nested lists containing the retweet and the user who retweeted it
    """
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    dfRT = df[['Usuario', 'Texto', 'Fecha']].copy()  # copied to work df
    idx = dfRT['Texto'].str.contains('RT @', na=False)
    dfRT = dfRT[idx]  # selecting rows with RT:@
    subset = dfRT[['Usuario', 'Texto']]  # date discarded
    retweetEdges = [list(x) for x in subset.to_numpy()]  # transform to list
    return retweetEdges

# Function to extract as list all the types of tweets, use specially for hashtag analysis (with get_edgesMain):

def get_all(df, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None):
    """
    Given a DataFrame containing all the tweets, the function returns all the tweet and the user who wrote or
    retweeted it in a nested list

    :param df: DataFrame with all the tweets
    :param keywords: List of words acting as key to filter the DataFrame
    :param stopwords: List of words destined to filter out the tweets that contain them
    :param keywords2: List of words acting as key to filter the DataFrame according to a subtopic
    :param stopwords2: List of words destined to filter out the tweets that contain them according to a subtopic
    :param interest: Active interest from the different categories available from the Lynguo tool
    :return: Nested lists containing tweet and user
    """
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    df_text = df[['Usuario', 'Texto']].copy()
    df_text = df_text.dropna()
    list_text = df_text['Texto'].to_numpy()
    return list_text

# Function to extract edges from RTs, mentions and all:

def get_edges(values):
    """
    Given a list of lists containing tweets or retweets and users the function returns the edges to create a network

    :param values: List of lists with the tweet and user
    :return: List of lists containing the user and the @ inside the tweet
    """
    edges = []
    for row in values:
        reg = re.search('@(\w+)', row[1])
        if reg:
            matchRT = reg.group(1)  # Se extrae la primera mención que hace referencia a la cuenta retuiteada
            # row[1] = hashlib.md5(match.encode()).hexdigest()
            row[1] = matchRT  # Convierte el nombre de la cuenta en hash y lo asigna al elemento
            edges.append(row)
    return edges


# Code to create a bar graph of most used hashtags in RTs:

# Selection of rows only with RTs and creation of a list containing the texts:


def get_hashtagsRT(df, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None):
    """
    Given a DataFrame containing all the tweets the function returns a list containing the different texts that are
    retweets in order to find the hashtags (#) inside them

    :param df: DataFrame with all the tweets
    :param keywords: List of words acting as key to filter the DataFrame
    :param stopwords: List of words destined to filter out the tweets that contain them
    :param keywords2: List of words acting as key to filter the DataFrame according to a subtopic
    :param stopwords2: List of words destined to filter out the tweets that contain them according to a subtopic
    :param interest: Active interest from the different categories available from the Lynguo tool
    :return: List with the retweets
    """
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    dfHashtagsRT = df[['Usuario', 'Texto']].copy()
    dfHashtagsRT = dfHashtagsRT.dropna()
    dfHashtagsRT = dfHashtagsRT[dfHashtagsRT['Texto'].str.match('RT @')]
    listHashtagsRT = dfHashtagsRT['Texto'].to_numpy()
    return listHashtagsRT

# Obtain hashtags used in Text:

def get_edgesHashRT(values):
    """
    Given a list containing retweets, the function finds all the hashtags inside the text
    :param values: list with the retweets
    :return: list with all the hashtags in these retweets
    """
    hashtags = []
    for row in values:
        match = re.findall('#(\w+)', row)
        for hashtag in match:
            hashtags.append(hashtag)
    return hashtags

# Organisation of hashtags by usage and creation of a list containing number of appearances:

def prepare_hashtags(hashtags, stopwords=None):
    """
    Given a list of hashtags, the function returns the number of appearances of each hashtags and a list of unique hashtags
    :param hashtags: list of hashtags
    :param stopwords: Word or list of words destined to be filtered out from the list of hashtags
    :return: Ordered list with the number of appearances of each hashtag and a list of unique hashtags
    """
    citsci_words = ['#citizenscience', 'citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana']
    hashtags = [x.lower() for x in hashtags]
    hashtags = [word for word in hashtags if word not in citsci_words]
    hashtags = [word for word in hashtags if word not in stopwords]
    hashtags = np.unique(hashtags, return_counts=True)
    hashtags = sorted((zip(hashtags[1], hashtags[0])), reverse=True)
    sortedNumberHashtags, sortedHashtagsRT = zip(*hashtags)
    return sortedNumberHashtags, sortedHashtagsRT


# Code to visualise the graph of hashtags in RTs:

def get_hashtagsRT2(df, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None):
    """
    Given a DataFrame containing all the tweets, the function returns a list of lists containing the retweets and the
    users who retweeted them, in order to find the hashtags inside those retweets

    :param df: DataFrame with all the tweets
    :param keywords: List of words acting as key to filter the DataFrame
    :param stopwords: List of words destined to filter out the tweets that contain them
    :param keywords2: List of words acting as key to filter the DataFrame according to a subtopic
    :param stopwords2: List of words destined to filter out the tweets that contain them according to a subtopic
    :param interest: Active interest from the different categories available from the Lynguo tool
    :return: List of lists, where each list contains user and retweet
    """
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    dfHashtagsRT = df[['Usuario', 'Texto']].copy()
    idx = dfHashtagsRT['Texto'].str.match('RT @', na=False)
    dfHashtagsRT = dfHashtagsRT[idx]
    listHashtagsRT = [list(x) for x in dfHashtagsRT.to_numpy()]
    return listHashtagsRT


def get_edgesHashRT2(values):
    """
    Given a list of list with users and retweets, the function returns the users and the hashtags in their retweets
    :param values: List of lists with user and retweet
    :return: List of lists, where each list contains user and hashtags
    """
    edges = []
    for row in values:
        match = re.search('#(\w+)', row[1])
        if match:
            matchHashRT = match.group(1)
            row[1] = matchHashRT
            edges.append(row)
    return edges

#Combination of edges of RTs and Mentions:


def combined_edges(x,y):
    """
    Given the edges from retweets and from mentions the function combine them both
    :param x: Edges from retweets
    :param y: Edges from mentions
    :return: List of lists with the edges combined
    """
    combined_edges = x + y
    return combined_edges

## Code to show the graph of related hashtags out of RTs. Shows hashtags related to each other


def get_hashtagsmain(df, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None):
    """
    Given a DataFrame containing all the tweets, the function returns a list containing all the mentions
    from the DataFrame

    :param df: DataFrame with all the tweets
    :param keywords: List of words acting as key to filter the DataFrame
    :param stopwords: List of words destined to filter out the tweets that contain them
    :param keywords2: List of words acting as key to filter the DataFrame according to a subtopic
    :param stopwords2: List of words destined to filter out the tweets that contain them according to a subtopic
    :param interest: Active interest from the different categories available from the Lynguo tool
    :return: List with all the tweets which are mentions
    """
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
    """

    :param values:
    :return:
    """
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

# Creation of the graph of most used hashtags (related to user):

def get_hashtagsmain2(df, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None):
    """
    Given a DataFrame with all the tweets, the function returns a list of list in which each list contains
    the user and the written tweet

    :param df: DataFrame with all the tweets
    :param keywords: List of words acting as key to filter the DataFrame
    :param stopwords: List of words destined to filter out the tweets that contain them
    :param keywords2: List of words acting as key to filter the DataFrame according to a subtopic
    :param stopwords2: List of words destined to filter out the tweets that contain them according to a subtopic
    :param interest: Active interest from the different categories available from the Lynguo tool
    :return: List of lists, each list contains user, written tweet
    """
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
    """
    Given a list of edges, the function returns the hashtags inside the tweet and relates them to the user

    :param values: List of lists containing the edges (user, tweet)
    :return: List of lists, in each list the user and the hashtag used by them is stored
    """
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

# Creation of the graph of most used hashtags out of RTs(use get_hashtagsmain()):

def get_edgesMain(values):
    """
    Given a list of tweets, the function returns the hashtags inside those tweets
    :param values: List of tweets
    :return: List with the hashtags inside the tweets
    """
    hashtags = []
    for row in values:
        match = re.findall('#(\w+)', row.lower())
        for hashtag in match:
            hashtags.append(hashtag)
    return hashtags

# Hashtags from the Bot:
botwords=['airpollution', 'luftdaten', 'fijnstof', 'waalre', 'pm2', 'pm10']

def prepare_hashtagsmain(list, stopwords=None):
    """
    Given a list of hashtags, the function returns the number of appearances of each hashtag and a unique list of hashtags
    :param list: List of hashtags
    :param stopwords: List of words destined to filter out the desired hashtags from the list
    :return: Ordered list with the number of appearances of each hashtag and a list of unique hashtags
    """
    citsci_words = ['#citizenscience', 'citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana']
    lista = [x.lower() for x in list]
    lista = [word for word in lista if word not in citsci_words]
    if stopwords != None:
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

# Function to extarct indegree, outdegree, eigenvector and betweenness stored in csv:

def csv_degval(Digraph, n, filename):
    """
    Given a Networkx directed graph, the function returns a CSV file containing the centrality measures of
    the graph (Indegree, Outdegree, Betweenness and Eigenvector) sorted by indegree

    :param Digraph: Networkx directed graph
    :param n: Number of users to store in the csv
    :param filename: Name for the CSV file
    :return: CSV file with the users, centrality measures and rank based of those measures, sorted by the indegree
    """
    list_values = []
    outdegrees2 = dict(Digraph.out_degree())
    indegrees = dict(Digraph.in_degree())
    centrality = dict(nx.eigenvector_centrality(Digraph))
    betweenness = dict(nx.betweenness_centrality(Digraph))
    indegtupl = sorted([(k, v) for k, v in indegrees.items()], key=lambda x:x[1], reverse=True)
    indegtupl = indegtupl[0:n]
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


def csv_Outdegval(Digraph, n, filename):
    """
    Given a Networkx directed graph, the function returns a CSV file containing the centrality measures of the
    graph (Indegree, Outdegree, Betweenness and Eigenvector) sorted by Outdegree

    :param Digraph: Networkx directed graph
    :param n: Number of users to store in the csv
    :param filename: Name for the CSV file
    :return: CSV file with the users, centrality measures and rank based of those measures, sorted by the Outdegree
    """
    outdegrees = dict(Digraph.out_degree())
    indegrees = dict(Digraph.in_degree())
    centrality = dict(nx.eigenvector_centrality(Digraph))
    betweenness = dict(nx.betweenness_centrality(Digraph))
    indegtupl = sorted([(k, v) for k, v in indegrees.items()], key=lambda x: x[1], reverse=True)

    outdegtupl = sorted([(k, v) for k, v in outdegrees.items()], key=lambda x: x[1], reverse=True)
    outdegtupl2 = outdegtupl[0:n]

    names = [i[0] for i in outdegtupl2]
    centraltupl = sorted([(k, v) for k, v in centrality.items()], key=lambda x: x[1], reverse=True)
    betwentupl = sorted([(k, v) for k, v in betweenness.items()], key=lambda x: x[1], reverse=True)

    list_values = []
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

    df.sort_values(by=['Outdegree'], ascending=False)

    return df.to_csv(filename, index=False)



## Functions to obtain elements for two mode:
# RTs:

def get_twomodeRT(df, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None):
    """
    Given a DataFrame containing al the tweets, the function returns a bipartite graph with the users and the
    retweets as nodes. The retweets are displayed as weighted nodes

    :param df: DataFrame with all the tweets
    :param keywords: List of words acting as key to filter the DataFrame
    :param stopwords: List of words destined to filter out the tweets that contain them
    :param keywords2: List of words acting as key to filter the DataFrame according to a subtopic
    :param stopwords2: List of words destined to filter out the tweets that contain them according to a subtopic
    :param interest: Active interest from the different categories available from the Lynguo tool
    :return: Networkx bipartite graph
    """
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    dfRT = df[['Usuario', 'Texto']].copy()
    idx = dfRT['Texto'].str.contains('RT @', na=False)
    dfRT = dfRT[idx]
    subset = dfRT[['Usuario', 'Texto']]
    subset = subset.drop_duplicates()
    u = list(subset['Usuario'])
    v = list(subset['Texto'])
    edges_tuple = [tuple(x) for x in subset.to_numpy()]
    G = nx.Graph()
    G.add_nodes_from(set(u), bipartite=0)
    G.add_nodes_from(set(v), bipartite=1)
    G.add_edges_from((x, y, {'weight': v}) for (x, y), v in Counter(edges_tuple).items())
    print(len(G.nodes))

    if len(G.nodes) >= 2000:
        G = nx.k_core(G, k=2)
    else:
        G = nx.k_core(G, k=1)

    counter = Counter(list((nx.core_number(G).values())))
    print(counter)
    pos = {}

    pos.update((node, (1, index)) for index, node in enumerate(set(u)))
    pos.update((node, (2, index)) for index, node in enumerate(set(v)))

    return G




# Obtaining components for two mode for hashtags outside RTs:

def get_twomodeHashMain(df, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None, filter_hashtags=None):
    """
    Given a DataFrame with all the tweets, the function returns a networkx bipartite graph with
    the users and hashtags (outside retweets) as nodes

    :param df: DataFrame with all the tweets
    :param keywords: List of words acting as key to filter the DataFrame
    :param stopwords: List of words destined to filter out the tweets that contain them
    :param keywords2: List of words acting as key to filter the DataFrame according to a subtopic
    :param stopwords2: List of words destined to filter out the tweets that contain them according to a subtopic
    :param interest: Active interest from the different categories available from the Lynguo tool
    :param filter_hashtags: Boolean, to remove the predefined citizen science most common hashtags
    :return: Networkx bipartite graph
    """
    edges = []
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    dfMain = df[['Usuario', 'Texto']].copy()
    dfMain = dfMain.dropna()
    dfEliminarRTs = dfMain[dfMain['Texto'].str.match('RT @')]
    dfMain = dfMain.drop(dfEliminarRTs.index)
    subset = dfMain[['Usuario', 'Texto']]
    subset = subset.drop_duplicates()
    listHT = [list(x) for x in subset.to_numpy()]
    stop_words = ['CitizenScience ','citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana','CitizenScience']

    filter_edges = []
    u = []
    v = []
    for row in listHT:
        match = re.search('#(\w+)', row[1])
        if match:
            matchhash = match.group(1)
            row[1] = matchhash
            edges.append(row)
    if filter_hashtags == True:
        for edge in edges:
            stop = False
            for word in edge:
                # print(word, word.lower() in stop_words)
                if word.lower() in stop_words:
                    stop = True
            if not stop:
                filter_edges.append(edge)

    if filter_hashtags == True:
        u = [x[0] for x in filter_edges]
        v = [x[1] for x in filter_edges]
        edges_tuple = [tuple(x) for x in filter_edges]
    else:
        u = [x[0] for x in edges]
        v = [x[1] for x in edges]
        edges_tuple = [tuple(x) for x in edges]

    G = nx.Graph()
    G.add_nodes_from(set(u), bipartite=0)
    G.add_nodes_from(set(v), bipartite=1)
    G.add_edges_from((x, y, {'weight': v}) for (x, y), v in Counter(edges_tuple).items())
    print(len(G.nodes))
    G.remove_edges_from(nx.selfloop_edges(G))

    if len(G.nodes) >= 2000:
        G = nx.k_core(G, k=2)
    else:
        G = nx.k_core(G, k=1)

    counter = Counter(list((nx.core_number(G).values())))
    print(counter)
    pos = {}

    pos.update((node, (1, index)) for index, node in enumerate(set(u)))
    pos.update((node, (2, index)) for index, node in enumerate(set(v)))

    return G

# Function to obtain the components for the two mode for hashtags in RTs:

def get_twomodeHashRT(df, keywords=None, stopwords=None, keywords2=None, stopwords2=None,
                      interest=None, filter_hashtags=None):
    """
     Given a DataFrame with all the tweets, the function returns a networkx bipartite graph with the
     users and hashtags (inside retweets) as nodes

    :param df: DataFrame with all the tweets
    :param keywords: List of words acting as key to filter the DataFrame
    :param stopwords: List of words destined to filter out the tweets that contain them
    :param keywords2: List of words acting as key to filter the DataFrame according to a subtopic
    :param stopwords2: List of words destined to filter out the tweets that contain them according to a subtopic
    :param interest: Active interest from the different categories available from the Lynguo tool
    :param filter_hashtags: Boolean, to remove the predefined citizen science most common hashtags
    :return: Networkx bipartite graph
    """
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    df = df[['Usuario', 'Texto']].copy()
    df = df.dropna()
    idx = df['Texto'].str.contains('RT @', na=False)
    dfRT = df[idx]  # Se seleccionan sólo las filas conpython  RT
    subset = dfRT[['Usuario', 'Texto']]
    listHT = [list(x) for x in subset.to_numpy()]
    edges = []
    stop_words = ['CitizenScience', 'citizenScience','citizenscience', 'rt', 'citizen', 'science',
                  'citsci', 'cienciaciudadana', '#CitizenScience']
    filter_edges = []
    u = []
    v = []
    for row in listHT:
        match = re.search('#(\w+)', row[1])
        if match:
            matchhash = match.group(1)
            row[1] = matchhash
            edges.append(row)
    if filter_hashtags == True:
        for edge in edges:
           stop = False
           for word in edge:
               #print(word, word.lower() in stop_words)
                if word.lower() in stop_words:
                    stop = True
           if not stop:
               filter_edges.append(edge)

    if filter_hashtags == True:
        u = [x[0] for x in filter_edges]
        v = [x[1] for x in filter_edges]
        edges_tuple = [tuple(x) for x in filter_edges]
    else:
        u = [x[0] for x in edges]
        v = [x[1] for x in edges]
        edges_tuple = [tuple(x) for x in edges]

    G = nx.Graph()
    G.add_nodes_from(set(u), bipartite=0)
    G.add_nodes_from(set(v), bipartite=1)
    G.add_edges_from((x, y, {'weight': v}) for (x, y), v in Counter(edges_tuple).items())
    print(len(G.nodes))
    G.remove_edges_from(nx.selfloop_edges(G))

    if len(G.nodes) >= 2000:
        G = nx.k_core(G, k=2)
    else:
        G = nx.k_core(G, k=1)

    counter = Counter(list((nx.core_number(G).values())))
    print(counter)
    pos = {}

    pos.update((node, (1, index)) for index, node in enumerate(set(u)))
    pos.update((node, (2, index)) for index, node in enumerate(set(v)))

    return G


# Wordcloud function for main hashtags:

def wordcloudmain(df, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None ):
    """
    Given a DataFrame containing all the tweets, the function returns a wordcloud with the hashtags
    (outside retweets) displayed by frequency

    :param df: DataFrame with all the tweets
    :param keywords: List of words acting as key to filter the DataFrame
    :param stopwords: List of words destined to filter out the tweets that contain them
    :param keywords2: List of words acting as key to filter the DataFrame according to a subtopic
    :param stopwords2: List of words destined to filter out the tweets that contain them according to a subtopic
    :param interest: Active interest from the different categories available from the Lynguo tool
    :return: Wordcloud with the hashtags by frequency
    """
    hashtags = []
    stop_words = ['citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana', 'CitizenScience']
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    df = df[['Usuario', 'Texto']].copy()
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


def wordcloud_mainhtlogo(df, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None, image=None):
    """
    Given a DataFrame containing all the tweets, the function returns a wordcloud with the hashtags
    (outside retweets) displayed by frequency

    :param df: DataFrame with all the tweets
    :param keywords: List of words acting as key to filter the DataFrame
    :param stopwords: List of words destined to filter out the tweets that contain them
    :param keywords2: List of words acting as key to filter the DataFrame according to a subtopic
    :param stopwords2: List of words destined to filter out the tweets that contain them according to a subtopic
    :param interest: Active interest from the different categories available from the Lynguo tool
    :param image: Image file to plot the wordcloud inside
    :return: Wordcloud inside desired image with the hashtags by frequency
    """
    hashtags =[]
    stop_words = ['citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana', 'CitizenScience']
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    df = df[['Usuario', 'Texto']].copy()
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

def wordcloudRT(df, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None ):
    """
    Given a DataFrame containing all the tweets, the function returns a wordcloud with the hashtags
    (inside retweets) displayed by frequency

    :param df: DataFrame with all the tweets
    :param keywords: List of words acting as key to filter the DataFrame
    :param stopwords: List of words destined to filter out the tweets that contain them
    :param keywords2: List of words acting as key to filter the DataFrame according to a subtopic
    :param stopwords2: List of words destined to filter out the tweets that contain them according to a subtopic
    :param interest: Active interest from the different categories available from the Lynguo tool
    :return: Wordcloud with the hashtags by frequency
    """
    hashtags =[]
    stop_words = ['citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana', 'CitizenScience']
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    df = df[['Usuario', 'Texto']].copy()
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

def wordcloudRT_logo(df, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None, image=None):
    """
    Given a DataFrame containing all the tweets, the function returns a wordcloud with the hashtags
    (inside retweets) displayed by frequency

    :param df: DataFrame with all the tweets
    :param keywords: List of words acting as key to filter the DataFrame
    :param stopwords: List of words destined to filter out the tweets that contain them
    :param keywords2: List of words acting as key to filter the DataFrame according to a subtopic
    :param stopwords2: List of words destined to filter out the tweets that contain them according to a subtopic
    :param interest: Active interest from the different categories available from the Lynguo tool
    :param image: Image file to plot the wordcloud inside
    :return: Wordcloud inside desired image with the hashtags by frequency
    """
    hashtags = []
    stop_words = ['citizenscience', 'rt', 'citizen', 'science', 'citsci', 'cienciaciudadana', 'CitizenScience']
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    df = df[['Usuario', 'Texto']].copy()
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


# Calculation of most used words:
# The function uses the column Text, we can select the number of words plotted:

def most_common(df, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None):
    """
    Given a DataFrame containing all the tweets, the function returns a dictionary with the most used words and
    the number of appearances, a list of these words and a list with the number of times these words appear

    :param df: DataFrame with all the tweets
    :param keywords: List of words acting as key to filter the DataFrame
    :param stopwords: List of words destined to filter out the tweets that contain them
    :param keywords2: List of words acting as key to filter the DataFrame according to a subtopic
    :param stopwords2: List of words destined to filter out the tweets that contain them according to a subtopic
    :param interest: Active interest from the different categories available from the Lynguo tool
    :return: Tuples dict containing word and number of times, list with the words and list with the times these words appear
    """
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    df = df[['Usuario', 'Texto']].copy()
    df = df.dropna()
    subset = df['Texto']
    subset = subset.dropna()
    words = " ".join(subset).lower().split()
    token = pos_tag(words, tagset='universal', lang='eng')
    word_list = [t[0] for t in token if (t[1] == 'NOUN', t[1] == 'VERB', t[1] == 'ADJ')]

    words = []
    for word in word_list:
        match = re.findall("\A[a-z-A-Z]+", word)
        for object in match:
            words.append(word)

    regex = re.compile(r'htt(\w+)')

    words = [word for word in words if not regex.match(word)]

    count_word = Counter(words)

    s = stopwords.words('english')
    e = stopwords.words('spanish')
    r = STOPWORDS
    d = stopwords.words('german')
    p = string.punctuation
    new_elements = (
    '\\n', 'rt', '?', '¿', '&', 'that?s', '??', '-', 'the', 'to', 'co', 'n', 'https', 'we?re', 'everyone?s',
    'supporters?', 'z', 'here:', 'science,', 'project.', 'citizen', 'science', 'us', 'student?', 'centre?', 'science?',
    ')', 'media?)', 'education?', 'reuse,', 'older!', 'scientists?', 'don?t', 'it?s', 'i?m', 'w/', 'w', 'more:')
    s.extend(new_elements)
    s.extend(e)
    s.extend(r)
    s.extend(d)
    s.extend(p)
    stopset = set(s)

    for word in stopset:
        del count_word[word]

    tuples_dict = sorted(count_word.items(), key=lambda x: x[1], reverse=True)
    words_pt = []
    numbers_pt = []

    for tuple in tuples_dict:
        words_pt.append(tuple[0])
        numbers_pt.append(tuple[1])

    return tuples_dict, words_pt, numbers_pt

# Top most used words in wordcloud:

def most_commonwc(df, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None):
    """
    Given a DataFrame containing all the tweets, the function returns a wordcloud with the most used words in these tweets

    :param df: DataFrame with all the tweets
    :param keywords: List of words acting as key to filter the DataFrame
    :param stopwords: List of words destined to filter out the tweets that contain them
    :param keywords2: List of words acting as key to filter the DataFrame according to a subtopic
    :param stopwords2: List of words destined to filter out the tweets that contain them according to a subtopic
    :param interest: Active interest from the different categories available from the Lynguo tool
    :return: Wordcloud with the most used words displayed in it
    """
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    df = df[['Usuario', 'Texto']].copy()
    df = df.dropna()
    subset = df['Texto']
    subset = subset.dropna()
    words = " ".join(subset).lower().split()
    token = pos_tag(words, tagset='universal', lang='eng')
    word_list = [t[0] for t in token if (t[1] == 'NOUN', t[1] == 'VERB', t[1] == 'ADJ')]

    words = []
    for word in word_list:
        match = re.findall("\A[a-z-A-Z]+", word)
        for object in match:
            words.append(word)

    regex = re.compile(r'htt(\w+)')

    words = [word for word in words if not regex.match(word)]

    count_word = Counter(words)

    s = stopwords.words('english')
    e = stopwords.words('spanish')
    r = STOPWORDS
    d = stopwords.words('german')
    p = string.punctuation
    new_elements = (
    '\\n', 'rt', '?', '¿', '&', 'that?s', '??', '-', 'the', 'to', 'co', 'n', 'https', 'we?re', 'everyone?s',
    'supporters?', 'z', 'here:', 'science,', 'project.', 'citizen', 'science', 'us', 'student?', 'centre?', 'science?',
    ')', 'media?)', 'education?', 'reuse,', 'older!', 'scientists?', 'don?t', 'it?s', 'i?m', 'w/', 'w', 'more:')
    s.extend(new_elements)
    s.extend(e)
    s.extend(r)
    s.extend(d)
    s.extend(p)
    stopset = set(s)

    for word in stopset:
        del count_word[word]

    wordcloud = WordCloud(width=900, height=900, background_color='white', stopwords=stopset,
                          min_font_size=10, max_words=300, collocations=False,
                          colormap='winter').generate_from_frequencies(count_word)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show

# Temporal series
# Selection of days between RTs or main texts:

def main_or_RT_days(df, RT=None):
    """
    Given a DataFrame with all the tweets, the function extracts all the dates for retweets or main tweets :param df:
    DataFrame with all the tweets

    :param RT: Boolean, whether to obtain the dates for retweets or main tweets
    :return: Subset containing the user, retweets or main tweets and formatted dates. Pandas series containing the
    sorted dates
    """
    df = df[['Fecha', 'Usuario', 'Texto']].copy()
    df = df.dropna()
    if RT == True:
        idx = df['Texto'].str.contains('RT @', na=False)
        subset = df[idx]
    if RT == False:
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

# Selection of days of both types of tweets, use with get_all():

def all_days(df):
    """
    Given a DataFrame containing all the tweets, the function extracts all the dates of these tweets

    :param df: DataFrame containing the tweets
    :return: Subset containing the user, tweets and formatted dates. Pandas series containing the sorted dates
    """
    df = df[['Fecha', 'Usuario', 'Texto']].copy()
    subset = df.dropna()
    subset['Fecha'] = pd.to_datetime(subset['Fecha'], errors='coerce')
    subset = subset.dropna()
    subset['Fecha'] = subset['Fecha'].dt.date

    # Obtenemos los días en el subset:
    df_Fecha = subset['Fecha']
    days = pd.unique(df_Fecha)
    days.sort()

    return subset, days


# Plot temporal series of hashtags usage through time. use Maindf o dfRT obtained with main_or_RT_days.
# Days: days obtained with previous function. Elements is the list of hashtags
# sortedMH (main hashtags) or sortedHT (RT) obtained with listHT/listHRT- get_edgesMain/
# get_EdgesHashRT- preparehashtagsmain/preparehashtags:

def plottemporalserie(days, df, elements, title=None, x=None, y=None):
    """
    Given a list of dates, a DataFrame containing all the tweets and a list of hashtags the function returns a
    figure representing the temporal evolution of use of these hashtags

    :param days: Pandas series of dates
    :param df: DataFrame containing the tweets
    :param elements: List of hashtags obtained from a filtered DataFrame
    :param title: Optional, title for the figure
    :param x: Number, position in list of the hashtag to plot first in the figure
    :param y: Number, last hashtag position desired to plot in the figure
    :return: Figure representing the use of hashtags through the time
    """
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
    plt.xlabel("Date", fontsize=15)
    plt.ylabel("n times", fontsize=15)
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", borderaxespad=0)

    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()

# plot temporal series of one hashtag of our interest (variable name):

def one_hastag_temporalseries(df, elements, days, name, title=None):
    """
    Given a list of dates, a DataFrame containing all the tweets and a list of hashtags the function returns a figure
    representing the temporal evolution of use of a desired hashtag

    :param df: DataFrame containing the tweets
    :param elements: List of hashtags obtained from a filtered DataFrame
    :param days: Pandas series of dates
    :param name: Name of the hashtag to plot
    :param title: Optional, title for the figure
    :return: Figure representing the use of one hashtag through the time
    """
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
            plt.xlabel("Date", fontsize=15)
            plt.ylabel("n times", fontsize=15)
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

            fig.autofmt_xdate()
            plt.show()

    else:
        print(name + 'not in list')
start_time = time.time()

# Function to obatin the top 50 tweets with higher impact/opinion and users with higher impact/opinion:


def impact_opinionRT(filename, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None, Impact=None, Opinion=None, n=None):
    """
    Given a DataFrame containing all the tweets, this function returns a CSV containing the users with higher
    impact/opinion in the dataset according to the retweets, it also plots a bar chart with the top users
    in impact/opinion

    :param filename: Path to DataFrame
    :param keywords: List of words acting as key to filter the DataFrame
    :param stopwords: List of words destined to filter out the tweets that contain them
    :param keywords2: List of words acting as key to filter the DataFrame according to a subtopic
    :param stopwords2: List of words destined to filter out the tweets that contain them according to a subtopic
    :param interest: Active interest from the different categories available from the Lynguo tool
    :param Impact: Boolean, whether to select impact or not
    :param Opinion: Boolean, wheter to select opinion or not
    :param n: Number of users to plot in the bar chart
    :return: CSV with the users sorted by impact/opinion. Bar chart with the top users by impact/opinion
    """
    df = pd.read_csv(filename, sep=';', encoding='utf-8', error_bad_lines=False, decimal=',', dtype={'Impacto':'float64'})
    df = df[['Texto', 'Usuario', 'Opinion', 'Impacto']].copy()
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
    """
    Given a DataFrame containing all the tweets, this function returns a CSV containing the users with higher
    impact/opinion in the dataset outside the retweets, it also plots a bar chart with the top users in impact/opinion

    :param filename: Path to DataFrame
    :param keywords: List of words acting as key to filter the DataFrame
    :param stopwords: List of words destined to filter out the tweets that contain them
    :param keywords2: List of words acting as key to filter the DataFrame according to a subtopic
    :param stopwords2: List of words destined to filter out the tweets that contain them according to a subtopic
    :param interest: Active interest from the different categories available from the Lynguo tool
    :param Impact: Boolean, whether to select impact or not
    :param Opinion: Boolean, whether to select opinion or not
    :param n: Number of users to plot in the bar chart
    :return: CSV with the users sorted by impact/opinion. Bar chart with the top users by impact/opinion
    """
    df = pd.read_csv(filename, sep=';', encoding='utf-8', error_bad_lines=False, decimal=',',
                     dtype={'Impacto': 'float64'})
    df = df[['Texto', 'Usuario', 'Opinion', 'Impacto']].copy()
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

# Sentiment analysis using Vader:

analyser = SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(sentence):
    """
    Vader sentiment analyser for a sentence
    :param sentence: A text
    :return: Score of sentiment for the text
    """""
    score = analyser.polarity_scores(sentence)
    return score

def sentiment_analyser(df,keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None):
    """
    Given a DataFrame containing all the tweets,the function returns a CSV containing the user and the score
    for each tweet from Vader sentiment

    :param df: A DataFrame containing all the tweets
    :param keywords: List of words acting as key to filter the DataFrame
    :param stopwords: List of words destined to filter out the tweets that contain them
    :param keywords2: List of words acting as key to filter the DataFrame according to a subtopic
    :param stopwords2: List of words destined to filter out the tweets that contain them according to a subtopic
    :param interest: Active interest from the different categories available from the Lynguo tool
    :return: CSV with the columns User, Text and Sentiment
    """
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    df = df[['Texto', 'Usuario']].copy()
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

# Plotbar and csv with tweets and users with higher sentiment, read the general sentiment analysis CSV previously created:

def sentiment_resultsRT(filename, n=None):
    """
    Given the sentiment analysis CSV created with VaderSentiment, this function returns a barplot with the top users
    in sentiment inside the retweets. It also stores a CSV with the top 50 retweets in sentiment

    :param filename: Path to the VaderSentiment CSV
    :param n: Number of users to plot in the chart
    :return: Barchart with the top users in sentiment. CSV with the top 50 retweets in sentiment
    """
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
    """
    Given the VaderSentiment CSV, this function returns a barplot with the top users in sentiment inside the retweets.
    It also stores a CSV with the top 50 tweets in sentiment

    :param filename: Path to the VaderSentiment CSV
    :param n: Number of users to plot in the chart
    :return: Barchart with the top users in sentiment. CSV with the top 50 tweets in sentiment
    """
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

# Combination of both subsets and ultimate calculation of users with higher Sentiment by Vader:

def combined_vader(subset1, subset2, n=None):
    """
    Given the subsets from RT Vadersentiment analysis and main tweets Vadersentiment analysis, this function combines
    both datasets and returns a barchart with the top users between the two types of tweets

    :param subset1: RTs subset
    :param subset2: Main tweets subset
    :param n: Number of users to plot in the figure
    :return: Barchart with the top desired number of users in Vader Sentiment
    """
    frames = [subset1, subset2]
    vader_df = pd.concat(frames, axis=0)
    vader_df = vader_df.groupby('User', as_index=False).mean()
    vader_df = vader_df.sort_values('Values', ascending=False)
    users = vader_df['User']
    values = vader_df['Values']
    plotbarchart(n, users, values, 'Top' + str(n) + 'Users with higher Sentiment', 'User', 'Sentiment')

# Weight adittion to edges list and DIGraph creation:
# First function, weight adittion as dict:

def make_weightedDiGraph(edges):
    """
    Given a list of edges, this function returns a networkx directed graph with the weighted edges

    :param edges: List of edges
    :return: Networkx directed graph (DiGraph) with the weighted edges calculated
    """
    edges_tuple = [tuple(x) for x in edges]
    G = nx.DiGraph((x, y, {'weight': v}) for (x, y), v in Counter(edges_tuple).items())
    return G

# Second function to add weight as third element in tuple (vertex, vertex, weight):

def weighted_DiGraph(edges):
    """
    Given a list of edges, this function returns a networkx directed graph with the weighted edges

    :param edges: List of edges
    :return: Networkx directed graph (Digraph) with the weighted edges calculated
    """
    for element in edges:
        element.append(1)

    result = Counter()

    for k, v, z in edges:
        result.update({(k,v):z})

    result = dict(result)

    tuple_edges_weight = []

    for key, value in result.items():
        temp = [key, value]
        tuple_edges_weight.append(temp)

    edges_tuple = []

    weights = []
    for element in tuple_edges_weight:
        edges_tuple.append(element[0])
        weights.append(element[1])

    edges_list = [list(x) for x in edges_tuple]

    for num in range(len(weights)):
        edges_list[num].append(weights[num])

    edges_weights = [tuple(x) for x in edges_list]

    G = nx.DiGraph()
    G.add_weighted_edges_from(edges_weights)
    return G

# Weight adittion to Graphs:

def weight_Graph(edges):
    """
    Given a list of edges, this function returns a networkx graph with the weighted edges

    :param edges: List of edges
    :return: Networkx graph with the weighted edges calculated
    """
    edges_tuple = [tuple(x) for x in edges]
    G = nx.Graph((x, y, {'weight': v}) for (x, y), v in Counter(edges_tuple).items())
    return G

# DF wit the calculation of mean, median, and sd of Impact and Opinion:

def dataframe_statistics(df, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None):
    """
    Given a DataFrame containing all the tweets, this function returns a new DataFrame containing the main statistical
    calculations around the original DataFrame

    :param df: DataFrame with all the tweets
    :param keywords: List of words acting as key to filter the DataFrame
    :param stopwords: List of words destined to filter out the tweets that contain them
    :param keywords2: List of words acting as key to filter the DataFrame according to a subtopic
    :param stopwords2: List of words destined to filter out the tweets that contain them according to a subtopic
    :param interest: Active interest from the different categories available from the Lynguo tool
    :return: DataFrame with the main statistical calculations from the DataFrame with tweets
    """
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    df = df[['Opinion', 'Impacto']].copy()
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

# Creation of DF containing structural analysis of a directed graph:


def graph_structural_analysis(Graph):
    """
    Given a Networkx Graph, this function returns a DataFrame which contains the main statistical calculations
    (number of nodes, number of edges, density, average shortest path, average path, clustering,
    transitivity and diameter) from the Network

    :param Graph: Networkx Graph
    :return: DataFrame with the statistical calculations
    """
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
    """
    Given a networkx graph with weighted nodes, this function returns a scatterplot with the distribution of weights

    :param Graph: Networkx Graph
    :return: Scatterplot with weight distribution
    """
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



# TFIDF wordcloud:

def tfidf_wordcloud(df, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None):
    """
    Given a DataFrame containing all the tweets, this function returns a wordcloud with the most important words,
    calculated with the TF-IDF method, displayed by frequency

    :param df: DataFrame with all the tweets
    :param keywords: List of words acting as key to filter the DataFrame
    :param stopwords: List of words destined to filter out the tweets that contain them
    :param keywords2: List of words acting as key to filter the DataFrame according to a subtopic
    :param stopwords2: List of words destined to filter out the tweets that contain them according to a subtopic
    :param interest: Active interest from the different categories available from the Lynguo tool
    :return: Wordcloud with the most relevant words according to TF-IDF calculation
    """
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    df_Text = df['Texto']
    df_Text = df_Text.dropna()
    df_Text = df_Text.drop_duplicates()
    tvec = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
    tvec_freq = tvec.fit_transform(df_Text.dropna())
    freqs = np.asarray(tvec_freq.mean(axis=0)).ravel().tolist()
    weights_df = pd.DataFrame({'term': tvec.get_feature_names(), 'freqs': freqs})
    weights_df = weights_df.sort_values(by='freqs', ascending=False)
    terms_list = list(weights_df['term'])
    terms_df = pd.DataFrame({'terms': terms_list})
    terms_df['terms'] = terms_df['terms'].apply(nltk.word_tokenize)

    unique_string = (' ').join(terms_list)
    token = word_tokenize(unique_string)
    token = pos_tag(token, tagset='universal', lang='eng')
    nouns = [t[0] for t in token if (t[1] == 'NOUN')]

    words = []
    for word in nouns:
        match = re.findall("\A[a-z-A-Z]+", word)
        for object in match:
            words.append(word)

    regex = re.compile(r'htt(\w+)')

    words = [word for word in words if not regex.match(word)]

    weights_df = weights_df[weights_df['term'].isin(words)]

    freqs = list(weights_df['freqs'])
    names = list(weights_df['term'])

    tagged = nltk.pos_tag(names)

    inverted_freqs = list(abs(np.log(freqs)))

    d = {}
    for key in names:
        for value in inverted_freqs:
            d[key] = value
            inverted_freqs.remove(value)
            break

    unique_string2 = (' ').join(names)

    wordcloud = WordCloud(width=900, height=900, background_color='white', min_font_size=10, max_words=400,
                          collocations=False, colormap='Pastel2')
    wordcloud.generate_from_frequencies(frequencies=d)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


# k core graphs:

def kcore_Graph(df, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None):
    """
    Given a DataFrame with all the tweets, this function returns a trimmed networkx graph according to the k-core
    decomposition of graphs

    :param df: DataFrame with all the tweets
    :param keywords: List of words acting as key to filter the DataFrame
    :param stopwords: List of words destined to filter out the tweets that contain them
    :param keywords2: List of words acting as key to filter the DataFrame according to a subtopic
    :param stopwords2: List of words destined to filter out the tweets that contain them according to a subtopic
    :param interest: Active interest from the different categories available from the Lynguo tool
    :return: Networkx Graph trimmed by networkx k-core algorithm
    """
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    dfRT = df[['Usuario', 'Texto']]
    idx = dfRT['Texto'].str.contains('RT @', na=False)
    dfRT = dfRT[idx]
    rt_edges_list = [list(x) for x in dfRT.to_numpy()]

    edges = []
    for row in rt_edges_list:
        reg = re.search('@(\w+)', row[1])
        if reg:
            matchRT = reg.group(1)
            row[1] = matchRT
            row[1] = hashlib.md5(matchRT.encode()).hexdigest()
            edges.append(row)

    G = make_weightedDiGraph(edges)
    G.remove_edges_from(nx.selfloop_edges(G))

    if len(G.nodes) >= 2000:
        G = nx.k_core(G, k=2)
    else:
        G = nx.k_core(G, k=1)

    core_number = nx.core_number(G)
    values = list(core_number.values())
    degree_count = Counter(values)
    print(len(G.nodes))

    return G

# Add an atribute to nodes in bipartite graph:

def add_attribute(df, category=None):
    """
    Given a DataFrame with the tweets, users and a category associated to a topic, this function returns a weighted
    networkx graph with the attribute from the category added to the nodes

    :param df: DataFrame with the tweets, users and category
    :param category: Name of the column category associated to a topic
    :return: Weighted networkx graph with the attribute added to the nodes
    """
    df = df[['Usuario', 'Texto', category]]
    df = df.dropna()
    idx = df['Texto'].str.contains('RT @', na=False)
    df = df[idx]

    G = nx.Graph()

    u = list(df['Usuario'])
    v = list(df['Texto'])
    subset = df[['Usuario', 'Texto']]
    print(len(set(u)))
    print(len(set(v)))
    edges_tuple = [tuple(x) for x in subset.to_numpy()]

    G.add_nodes_from(set(u), bipartite=0)
    G.add_nodes_from(set(v), bipartite=1)

    for index, row in df.iterrows():
        G.nodes[row['Texto']][category] = row[category]

    G.add_edges_from((x, y, {'weight': v}) for (x, y), v in Counter(edges_tuple).items())
    return G


