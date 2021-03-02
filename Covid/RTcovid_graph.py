import sys
sys.path.insert(1, '/home/fernan/Documents/Proyectos/CSTrack-URJC')
import utils
import networkx as nx
from webweb import Web

covid = ['covid', 'coronavirus', 'pandemic', 'COVID-19', 'SARS-COV-2', 'vaccine', 'vaccination']

rtlist = utils.get_retweets('/home/fernan/Documents/Lynguo_def2.csv', keywords=covid)

rtedges = utils.get_edges(rtlist)

G = nx.Graph()
G.add_edges_from(rtedges)
subgraphs = utils.get_subgraphs(G)
subgraphs = [graph for graph in subgraphs if len(graph.nodes) > 5]

web = Web(title="retweets", nx_G=subgraphs[0])
web.display.gravity = 1

name = "graph"
for i in range(2, len(subgraphs)):
    web.networks.retweets.add_layer(nx_G=subgraphs[i])

# show the visualization
web.show()