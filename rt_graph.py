import pandas as pd
import re
import networkx as nx
from webweb import Web


def get_subgraphs(graph):
    import networkx as nx
    components = list(nx.connected_components(graph))
    list_subgraphs = []
    for component in components:
        list_subgraphs.append(graph.subgraph(component))

    return list_subgraphs

df = pd.read_csv('lynguo2.csv', sep=';', error_bad_lines=False)
stopwords = ['#citizenscience', 'citizenscience', 'rt', 'citizen', 'science', 'citsci','cienciaciudadana']
#df = df.dropna()

# CALCULAR GRAFO RTs
dfRT = df[['Usuario', 'Texto', 'Fecha']].copy() # Se copia a un dataframe de trabajo

idx = dfRT['Texto'].str.contains('RT @', na=False)
dfRT = dfRT[idx]  # Se seleccionan sólo las filas con RT


subset = dfRT[['Usuario', 'Texto']] # Se descarta la fecha
print(subset)
retweetEdges = [list(x) for x in subset.to_numpy()] # Se transforma en una lista

for row in retweetEdges:
    reg = re.search('@(\w+)', row[1])
    if reg:
        matchRT = reg.group(1)  # Se extrae la primera mención que hace referencia a la cuenta retuiteada
        row[1] = matchRT  # Convierte el nombre de la cuenta en hash y lo asigna al elemento


dfCitas = pd.DataFrame(retweetEdges)
dfCitas.to_csv('retweetEdges.csv', header=False, index=False, sep=';')
G = nx.Graph()
G.add_edges_from(retweetEdges)
subgraphs = get_subgraphs(G)
web = Web(title="retweets", nx_G=subgraphs[0])
web.display.gravity = 1

name ="graph"
for i in range(1, len(subgraphs)):
    web.networks.retweets.add_layer(nx_G=subgraphs[i])

# show the visualization
web.show()

#grafoRT = graph_utils.creategraph(retweetEdges)
#graph_utils.plotgraph(grafoRT, 'grafoRT', False)