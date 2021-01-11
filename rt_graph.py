import pandas as pd
import re
import networkx as nx
from webweb import Web
import utils

retweetList = utils.get_retweets("sample.csv")
retweetEdges = utils.get_edges(retweetList)

dfCitas = pd.DataFrame(retweetEdges)
dfCitas.to_csv('retweetEdges.csv', header=False, index=False, sep=';')
G = nx.Graph()
G.add_edges_from(retweetEdges)
subgraphs = utils.get_subgraphs(G)
subgraphs = [graph for graph in subgraphs if len(graph.nodes) > 5]
web = Web(title="retweets", nx_G=subgraphs[1])
web.display.gravity = 1

name ="graph"
for i in range(2, len(subgraphs)):
    web.networks.retweets.add_layer(nx_G=subgraphs[i])

# show the visualization
web.show()

#grafoRT = graph_utils.creategraph(retweetEdges)
#graph_utils.plotgraph(grafoRT, 'grafoRT', False)