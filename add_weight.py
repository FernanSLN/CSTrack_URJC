import pandas as pd
import networkx as nx
from webweb import Web
import utils
import matplotlib.pyplot as plt
from collections import Counter

#retweetList = utils.get_retweets('/home/fernan/Documents/Lynguo_def2.csv')

#retweetEdges = utils.get_edges(retweetList)

ejes = [['Antonio', 'Pedro'], ['Antonio', 'Pedro'], ['Josefina', 'Doña Rojelia'], ['Eltio lavara', 'El cocas de Cruz y Raya']]

e = ['Antonio', 'Pedro']

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


G = weighted_graph(ejes)
nx.draw(G)
plt.show()


