import pandas as pd
import re
from webweb import Web


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
    matchRT = re.search('@(\w+)', row[1]).group(1)  # Se extrae la primera mención que hace referencia a la cuenta retuiteada
    row[1] = matchRT  # Convierte el nombre de la cuenta en hash y lo asigna al elemento

dfCitas = pd.DataFrame(retweetEdges)
dfCitas.to_csv('retweetEdges.csv', header=False, index=False, sep=';')

web = Web(retweetEdges)
web.display.gravity = 1

# show the visualization
web.show()

#grafoRT = graph_utils.creategraph(retweetEdges)
#graph_utils.plotgraph(grafoRT, 'grafoRT', False)