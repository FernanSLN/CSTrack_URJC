import sys
sys.path.insert(1, '/home/fernan/Documents/Proyectos/CSTrack-URJC')
import utils
import networkx as nx
from webweb import Web
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt


SDGS_marca = ['CS SDG c', 'CS SDG','CS SDG co', 'CS SDG conference', 'CS SDG conference 2020',
              'CS SDG conference 20', 'CS SDG confer', 'CS SDG conference ', 'CS SDG conferenc']

retweetList = utils.get_retweets("/home/fernan/Documents/Lynguo_def2.csv", interest=SDGS_marca)
retweetEdges = utils.get_edges(retweetList)
G = utils.weighted_graph(retweetEdges)
G = G.to_undirected()
partition = community_louvain.best_partition(G)
pos = nx.spring_layout(G)
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40, cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()
