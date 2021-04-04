import community as cl
import dash_utils
import pandas as pd
import matplotlib.cm as cm
import networkx as nx
import matplotlib.pyplot as plt
import datetime
import cugraph
import networkit as nk

df = pd.read_csv("Lynguo_def2.csv", sep=";", encoding="latin-1", error_bad_lines=False)
start = datetime.datetime.now()
G = dash_utils.get_graph_rt(df)
n_g = nk.nxadapter.nx2nk(G)
idmap = dict((u, id) for (id, u) in zip(G.nodes(), range(G.number_of_nodes())))
btwn = nk.centrality.Betweenness(n_g)
ec = nk.centrality.EigenvectorCentrality(n_g)
ec.run()
btwn.run()
nodes = n_g.iterNodes()
for node in nodes:
    print("In:", n_g.degreeIn(node))
    print("Out:", n_g.degreeOut(node))
"""print(idmap)
communities = nk.community.detectCommunities(n_g)
for i in range(0, communities.numberOfSubsets()):
    for member in communities.getMembers(i):
        print(idmap[member], end=",")
    print("---- COMUNITY -----")"""
print("END")
print(btwn.ranking()[:10])
print("EIGENVECTOR")
print(ec.ranking()[:10])
print("Time:", datetime.datetime.now() - start)