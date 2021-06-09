import sys
sys.path.insert(1, '/home/fernan/Documents/Proyectos/CSTrack-URJC')
import utils
import pandas as pd
import networkx as nx
from DataFrame import df
from sdgs_list import sdgs_keywords


rts = utils.get_retweets(df, keywords= sdgs_keywords)
edges = utils.get_edges(rts)
G = nx.DiGraph()
G.add_edges_from(edges)
utils.csv_degval(G, 'Degree data Special Issue')
utils.csv_Outdegval(G, 'Outdegree data Special Issue')
