from webweb import Web
import networkx as nx
import utils
import RTcovid_graph

citations_list = utils.get_cites('/home/fernan/Documents/Lynguo_def2.csv', keywords=RTcovid_graph.covid)

cite_edges = utils.get_edges(citations_list)

G = nx.Graph()
G.add_edges_from(cite_edges)
#subgraphs = utils.get_subgraphs(G)

 #subgraphs = [graph for graph in subgraphs if len(graph.nodes) > 5]
citas = Web(title="citas", nx_G=G)

#for i in range(1, len(subgraphs)):
    #citas.networks.citas.add_layer(nx_G=subgraphs[i])

citas.display.gravity=1
citas.show()

