import sys
sys.path.insert(1, '/home/fernan/Documents/Proyectos/CSTrack-URJC')
from utils import utils
import networkx as nx
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt


SDGS_marca = ['CS SDG c', 'CS SDG','CS SDG co', 'CS SDG conference', 'CS SDG conference 2020',
              'CS SDG conference 20', 'CS SDG confer', 'CS SDG conference ', 'CS SDG conferenc']

retweetList = utils.get_retweets("/home/fernan/Documents/Lynguo_def2.csv", interest=SDGS_marca)
retweetEdges = utils.get_edges(retweetList)
G = utils.weighted_graph(retweetEdges)
D = nx.to_undirected(G)
partition = community_louvain.best_partition(D)
pos = nx.spring_layout(D)
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(D, pos, partition.keys(), node_size=40, cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(D, pos, alpha=0.5)
plt.show()

print(nx.number_of_nodes(G))
print(nx.number_of_edges(G))
print(nx.density(G))
print('not weakly connected, average path length not computable')
print(nx.average_clustering(G))
print(nx.transitivity(G))
print('not weakly connected, diameter is infinite')
