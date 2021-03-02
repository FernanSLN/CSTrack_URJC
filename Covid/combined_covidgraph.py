import sys
sys.path.insert(1, '/home/fernan/Documents/Proyectos/CSTrack-URJC')
import utils
import RTcovid_graph
from webweb import Web
import networkx as nx

retweetList = utils.get_retweets("/home/fernan/Documents/Lynguo_def2.csv", keywords=RTcovid_graph.covid)
retweetEdges = utils.get_edges(retweetList)

mentionList = utils.get_cites("/home/fernan/Documents/Lynguo_def2.csv", keywords=RTcovid_graph.covid)
mentionEdges = utils.get_edges(mentionList)

combined_edges = utils.combined_edges(mentionEdges,retweetEdges)

G = nx.Graph()
G.add_edges_from(combined_edges)
subgraphs = utils.get_subgraphs(G)
subgraphs = [graph for graph in subgraphs if len(graph.nodes) > 5]

web = Web(title="Cmbined graph", nx_G=subgraphs[0])
web.display.gravity = 1

name ="graph"
for i in range(2, len(subgraphs)):
    web.networks.rtscites.add_layer(nx_G=subgraphs[i])

# show the visualization
web.show()