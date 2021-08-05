from webweb import Web
import networkx as nx
import utils

listHashtagsRT2 = utils.get_hashtagsRT2("/home/fernan/Documents/Lynguo_def2.csv")
edges2 = utils.get_edgesHashRT2(listHashtagsRT2)

G = nx.Graph()
G.add_edges_from(edges2)

# Empleando G.remove_nodes_from() podemos eliminar las stop_words para filtrar el grafo

# G.remove_nodes_from(utils.stop_words)

subgraphs = utils.get_subgraphs(G)
subgraphs = [graph for graph in subgraphs if len(graph.nodes) > 5]
web = Web(title="HashtagsRT", nx_G=subgraphs[0])
web.display.gravity = 1

name ="graph"
for i in range(2, len(subgraphs)):
    web.networks.hashrts.add_layer(nx_G=subgraphs[i])

# show the visualization
web.show()