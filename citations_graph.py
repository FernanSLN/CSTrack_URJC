import pandas as pd
import re
from webweb import Web
import hashlib
import networkx as nx
import matplotlib.pyplot as plt

df = pd.read_csv('Lynguo_def2.csv', sep= ';', error_bad_lines = False)

df= df.drop([78202], axis= 0)

dfMentions = df[['Usuario', 'Texto']].copy()

dfMentions=dfMentions.dropna()

dfEliminarRTs = dfMentions[dfMentions['Texto'].str.match('RT @')]

dfMentions=dfMentions.drop(dfEliminarRTs.index)

mentionsSubset = dfMentions[['Usuario', 'Texto']]

mentionsList = [list(x) for x in mentionsSubset.to_numpy()]

mentionEdges = []

for row in mentionsList:
    match = re.search('@(\w+)', row[1])
    if match:
        match = match.group(1)
        row[1] = hashlib.md5(match.encode()).hexdigest()
        mentionEdges.append(row)

web = Web(mentionEdges)

G = nx.Graph()

G.add_edges_from(mentionEdges)

lista_cc = list(nx.connected_components(G))

Gmax = max(nx.connected_components(G), key=len) #Extraemos el componente más conectado

def get_subgraphs(graph):
    import networkx as nx
    components = list(nx.connected_components(graph))
    list_subgraphs = []
    for component in components:
        list_subgraphs.append(graph.subgraph(component))

    return list_subgraphs

subgraphs = get_subgraphs(G)

print(len(get_subgraphs(G)))

web1= Web(nx_G=subgraphs[1])

web1.networks.add_layer(nx_G=subgraphs[2])

web1.networks.add_layer(nx_G=subgraphs[3])


citas = Web(title="citas", nx_G=subgraphs[0])

for i in range(1, len(subgraphs)):
    citas.networks.add(nx_G=subgraphs[i])

citas.display.gravity=1

citas.show()




