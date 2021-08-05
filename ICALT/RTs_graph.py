import pandas as pd
import networkx as nx
from webweb import Web
import utils
from ICALT.keywords_icalt import k, k_stop

retweetList = utils.get_retweets("/home/fernando/Documentos/Lynguo_def2.csv", k, k_stop)
print("LENGTH: ", len(retweetList))
retweetEdges = utils.get_edges(retweetList)

dfCitas = pd.DataFrame(retweetEdges)
dfCitas.to_csv('retweetEdges.csv', header=False, index=False, sep=';')
G = nx.Graph()
G.add_edges_from(retweetEdges)
subgraphs = utils.get_subgraphs(G)
subgraphs = [graph for graph in subgraphs if len(graph.nodes) > 5]
print("BIGGEST: ", len(subgraphs[0].nodes))
web = Web(title="retweets", nx_G=subgraphs[0])
web.display.gravity = 1

name = "graph"
for i in range(2, len(subgraphs)):
    web.networks.retweets.add_layer(nx_G=subgraphs[i])

# show the visualization
web.show()