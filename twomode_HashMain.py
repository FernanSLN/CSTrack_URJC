from utils import get_uv_HashMain, get_subgraphs, direct_subgraphs
import networkx as nx
from webweb import Web

edges, u, v = get_uv_HashMain('/home/fernan/Documents/Lynguo_def2.csv', 'education')

B = nx.Graph()

B.add_nodes_from(set(u), bipartite=0)

B.add_nodes_from(set(v), bipartite=1)

B.add_edges_from(edges)

# Graficado con Web

N = nx.Graph()

N.add_nodes_from(set(u), bipartite=0)

N.add_nodes_from(set(v), bipartite=1)

N.add_edges_from(edges)

pos = {}

pos.update((node, (1, index)) for index, node in enumerate(set(u)))
pos.update((node, (2, index)) for index, node in enumerate(set(v)))

subgraphs = get_subgraphs(N)

subgraphs = [graph for graph in subgraphs if len(graph.nodes) > 5]

disubgraphs = direct_subgraphs(subgraphs)

web = Web(title="main graph", nx_G=disubgraphs[0])
web.display.gravity = 1

name = "graph"
for i in range(2, len(disubgraphs)):
    web.networks.graphs.add_layer(nx_G=disubgraphs[i])

web.show()