import sys
sys.path.insert(1, '/home/fernan/Documents/Proyectos/CSTrack-URJC')
import utils
import pandas as pd
import networkx as nx
import DataFrame

with open("/home/fernan/Documents/Lista de SDGS.txt", "r") as file:
    lines = file.readlines()
    sdgs_keywords = []
    for l in lines:
        sdgs_keywords.append(l.replace("\n", ""))


rts = utils.get_retweets(DataFrame.df, keywords= sdgs_keywords)
edges = utils.get_edges(rts)
G = nx.DiGraph()
G.add_edges_from(edges)
utils.csv_degval(G,'Degree data Special Issue')