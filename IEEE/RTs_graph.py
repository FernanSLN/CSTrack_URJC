import sys
sys.path.append('/home/fernan/Documents/Proyectos/CSTrack-URJC')
from utils import utils
from modin_Dataframe import df
from sdgs_list import sdgs_keywords
from webweb import Web
import networkx as nx

G = utils.kcore_Graph(df, keywords=sdgs_keywords)
print(len(G.nodes))
print(len(G.edges))
print(nx.number_strongly_connected_components(G))
print(nx.number_weakly_connected_components(G))

H = nx.to_undirected(G)
subgraphs = utils.get_subgraphs(H)

print(len(subgraphs))

longitude = []
for subgraph in subgraphs:
    node_len = len(subgraph.nodes)
    longitude.append(node_len)

print(longitude)

web = Web(title="retweets", nx_G=subgraphs[0])
web.display.gravity = 1

# show the visualization
web.show()