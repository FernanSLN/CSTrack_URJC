import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import utils

def weighted_graph(ejes):
    for lista in ejes:
        lista.append(1)

    result = Counter()

    for k, v, z in ejes:
        result.update({(k,v):z})

    result = dict(result)

    tupla_ejes_peso = []

    for key, value in result.items():
        temp = [key, value]
        tupla_ejes_peso.append(temp)

    ejes_tupla = []

    pesos = []
    for lista in tupla_ejes_peso:
        ejes_tupla.append(lista[0])
        pesos.append(lista[1])

    ejes_lista = [list(x) for x in ejes_tupla]

    for num in range(len(pesos)):
        ejes_lista[num].append(pesos[num])

    ejes_pesos = [tuple(x) for x in ejes_lista]

    G = nx.DiGraph()
    G.add_weighted_edges_from(ejes_pesos)
    return G

retweetList = utils.get_retweets('/home/fernan/Documents/Lynguo_def2.csv')

edges = utils.get_edges(retweetList)

G = weighted_graph(edges)
nx.draw(G)
plt.show()


