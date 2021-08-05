from webweb import Web
import networkx as nx
import utils

hashmain = utils.get_hashtagsmain2("/home/fernan/Documents/Lynguo_def2.csv")
hashmain2= utils.get_edgesmain2(hashmain)
hashtags_edges = utils.prepare_hashtags2(hashmain2)

G = nx.Graph()
G.add_edges_from(hashtags_edges)

# Empleando G.remove_nodes_from() podemos eliminar las stop_words para filtrar el grafo

# G.remove_nodes_from(utils.stop_words)

#subgraphs = utils.get_subgraphs(G)
#subgraphs = [graph for graph in subgraphs if len(graph.nodes) > 5]

web = Web(title="Main Hashtags",nx_G=G)
#web = Web(title="Main Hashtags", nx_G=subgraphs[0])
web.display.gravity = 1

#for i in range(2, len(subgraphs)):
   # web.networks.hashtags.add_layer(nx_G=subgraphs[i])

# show the visualization
web.show()