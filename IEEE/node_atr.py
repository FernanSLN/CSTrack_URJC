import modin.pandas as mdpd
import ray
import networkx as nx
from collections import Counter
from webweb import Web

ray.init()

df = mdpd.read_csv("/home/fernan/Documents/tweets_with_category_bert.csv", sep=',', encoding='utf-8', decimal=',',
                 error_bad_lines=False)

df = df[['Usuario', 'Texto', 'SDG']]
df = df.dropna()
idx = df['Texto'].str.contains('RT @', na=False)
df = df[idx]

print(df)

G = nx.Graph()


print('LLega 1')

u = list(df['Usuario'])
v = list(df['Texto'])
subset = df[['Usuario', 'Texto']]

edges_tuple = [tuple(x) for x in subset.to_numpy()]

G.add_nodes_from(set(u), bipartite=0)
G.add_nodes_from(set(v), bipartite=1)

print('Llega 2')

for index, row in df.iterrows():
    G.nodes[row['Texto']]['SDG'] = row['SDG']

print('Llega 3')

G.add_edges_from((x, y, {'weight': v}) for (x, y), v in Counter(edges_tuple).items())

print('Lega 4')


# sacar k = 2,3 y 4 y su len nodes


G = nx.k_core(G, k=2)

print(len(G.nodes))

pos = {}

pos.update((node, (1, index)) for index, node in enumerate(set(u)))
pos.update((node, (2, index)) for index, node in enumerate(set(v)))

web = Web(title="K-core 2: graph of retweets", nx_G=G)
web.display.gravity = 1
web.display.sizeBy = 'bipartite'
web.display.colorBy = 'SDG'
web.display.charge = 40
web.display.linkLength = 10
web.display.linkStrength = 1
web.display.radius = 2.5
web.display.scaleLinkOpacity = True
web.display.scaleLinkWidth = True

# show the visualization
web.show()

H = nx.k_core(G, k=3)

print(len(H.nodes))

pos = {}

pos.update((node, (1, index)) for index, node in enumerate(set(u)))
pos.update((node, (2, index)) for index, node in enumerate(set(v)))

web2 = Web(title="K-core 3: graph of retweets", nx_G=H)
web2.display.gravity = 0.8
web2.display.gravity = 1
web2.display.sizeBy = 'bipartite'
web2.display.colorBy = 'SDG'
web2.display.charge = 40
web2.display.linkLength = 10
web2.display.linkStrength = 1
web2.display.radius = 2.5
web2.display.scaleLinkOpacity = True
web2.display.scaleLinkWidth = True

# show the visualization
web2.show()