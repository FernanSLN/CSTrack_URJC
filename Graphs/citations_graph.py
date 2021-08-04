from webweb import Web
import networkx as nx
import utils

mentionsList = utils.get_cites("/home/fernando/Documentos/Lynguo_def2.csv")
mentionEdges = utils.get_edges(mentionsList)

G = nx.Graph()
G.add_edges_from(mentionEdges)
subgraphs = utils.get_subgraphs(G)
#We only show graphs with more than 5 nodes
subgraphs = [graph for graph in subgraphs if len(graph.nodes) > 5]
citas = Web(title="citas", nx_G=subgraphs[0])

for i in range(1, len(subgraphs)):
    citas.networks.citas.add_layer(nx_G=subgraphs[i])

citas.display.gravity=1
citas.show()




