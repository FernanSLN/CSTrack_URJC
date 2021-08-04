import utils
import networkx as nx
from IEEE.modin_Dataframe import df
from IEEE.sdgs_list import sdgs_keywords


rts = utils.get_retweets(df, keywords= sdgs_keywords)
edges = utils.get_edges(rts)
G = nx.DiGraph()
G.add_edges_from(edges)
utils.csv_degval(G, 50, 'Degree data Special Issue')
utils.csv_Outdegval(G, 50, 'Outdegree data Special Issue')
