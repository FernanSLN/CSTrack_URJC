import networkx as nx
import utils
import community as community_louvain
import matplotlib.pyplot as plt
import matplotlib.cm as cm



hashmain = utils.get_hashtagsmain2('/home/fernan/Documents/Lynguo_def2.csv')
hashmain2 = utils.get_edgesmain2(hashmain)
hashtags_edges = utils.prepare_hashtags2(hashmain2)
G = nx.Graph()
G.add_edges_from(hashtags_edges)
partition = community_louvain.best_partition(G)

pos = nx.spring_layout(G)
# color the nodes according to their partition
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40, cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()