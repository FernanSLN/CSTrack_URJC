import networkx as nx
from webweb import Web
from utils import utils
import pandas as pd
import re
import hashlib
from collections import Counter

def kcore_Graph(df, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None):
    df = utils.filter_by_interest(df, interest)
    df = utils.filter_by_topic(df, keywords, stopwords)
    df = utils.filter_by_subtopic(df, keywords2, stopwords2)
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

    G = utils.make_weightedDiGraph(edges)
    G.remove_edges_from(nx.selfloop_edges(G))
    core_number = nx.core_number(G)
    values = list(core_number.values())
    degree_count = Counter(values)
    G_kcore = nx.k_core(G, k=2)
    print(len(G_kcore.nodes))
    G_kcore_undirected = nx.to_undirected(G_kcore)
    subgraphs = utils.get_subgraphs(G_kcore_undirected)
    subgraphs = [graph for graph in subgraphs if len(graph.nodes) > 5]
    subgraphs = utils.direct_subgraphs(subgraphs)

    return subgraphs

df = pd.read_csv('/home/fernan/Documents/Lynguo_April21.csv', sep=';', encoding='utf-8', error_bad_lines=False)
subgraphs = kcore_Graph(df)
web = Web(title="retweets", nx_G=subgraphs[0])
web.display.gravity = 1

for i in range(2, len(subgraphs)):
    web.networks.retweets.add_layer(nx_G=subgraphs[i])

# show the visualization
web.show()