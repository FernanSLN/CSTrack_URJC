{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import re\n",
    "from webweb import Web\n",
    "import hashlib\n",
    "import networkx as nx\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('Lynguo_def2.csv', sep= ';', error_bad_lines = False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "df= df.drop([78202], axis= 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "dfMentions = df[['Usuario', 'Texto']].copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "dfMentions=dfMentions.dropna()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "dfEliminarRTs = dfMentions[dfMentions['Texto'].str.match('RT @')]\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "dfMentions=dfMentions.drop(dfEliminarRTs.index)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "mentionsSubset = dfMentions[['Usuario', 'Texto']]\n",
    "\n",
    "mentionsList = [list(x) for x in mentionsSubset.to_numpy()]\n",
    "mentionEdges = []\n",
    "for row in mentionsList:\n",
    "    match = re.search('@(\\w+)', row[1])\n",
    "    if match:\n",
    "        match = match.group(1)\n",
    "        row[1] = hashlib.md5(match.encode()).hexdigest()  \n",
    "        mentionEdges.append(row)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "web = Web(mentionEdges)\n",
    "web.display.gravity = 1\n",
    "web.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "G = nx.Graph()\n",
    "G.add_edges_from(mentionEdges)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [],
   "source": [
    "lista_cc = list(nx.connected_components(G))\n",
    "Gmax = max(nx.connected_components(G), key=len) #Extraemos el componente más conectado"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_subgraphs(graph):\n",
    "    import networkx as nx\n",
    "    components = list(nx.connected_components(graph))\n",
    "    list_subgraphs = []\n",
    "    for component in components:\n",
    "        list_subgraphs.append(graph.subgraph(component))\n",
    "\n",
    "    return list_subgraphs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "subgraphs = get_subgraphs(G)\n",
    "web1 = Web(nx_G= nx.Graph(subgraphs[1]))\n",
    "web1.display.gravity = 1\n",
    "web1.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "3378\n"
     ]
    }
   ],
   "source": [
    "print(len(get_subgraphs(G)))\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "metadata": {},
   "outputs": [],
   "source": [
    "web1= Web(nx_G=subgraphs[1])\n",
    "web1.networks.add_layer(nx_G=subgraphs[2])\n",
    "web1.networks.add_layer(nx_G=subgraphs[3])\n",
    "web1.display.networkName = 'web1'\n",
    "web.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "H= G.subgraph(subgraphs[1])\n",
    "nx.draw(H)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "metadata": {},
   "outputs": [
    {
     "ename": "AttributeError",
     "evalue": "'set' object has no attribute 'edges'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-100-09af07c9c1a1>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mnx\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdraw\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mGmax\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32m~/anaconda3/lib/python3.8/site-packages/networkx/drawing/nx_pylab.py\u001b[0m in \u001b[0;36mdraw\u001b[0;34m(G, pos, ax, **kwds)\u001b[0m\n\u001b[1;32m    121\u001b[0m         \u001b[0mkwds\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m\"with_labels\"\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m\"labels\"\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mkwds\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    122\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 123\u001b[0;31m     \u001b[0mdraw_networkx\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mG\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mpos\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mpos\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0max\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0max\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwds\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    124\u001b[0m     \u001b[0max\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mset_axis_off\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    125\u001b[0m     \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdraw_if_interactive\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/anaconda3/lib/python3.8/site-packages/networkx/drawing/nx_pylab.py\u001b[0m in \u001b[0;36mdraw_networkx\u001b[0;34m(G, pos, arrows, with_labels, **kwds)\u001b[0m\n\u001b[1;32m    334\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    335\u001b[0m     \u001b[0mdraw_networkx_nodes\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mG\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mpos\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mnode_kwds\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 336\u001b[0;31m     \u001b[0mdraw_networkx_edges\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mG\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mpos\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0marrows\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0marrows\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0medge_kwds\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    337\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mwith_labels\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    338\u001b[0m         \u001b[0mdraw_networkx_labels\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mG\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mpos\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mlabel_kwds\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/anaconda3/lib/python3.8/site-packages/networkx/drawing/nx_pylab.py\u001b[0m in \u001b[0;36mdraw_networkx_edges\u001b[0;34m(G, pos, edgelist, width, edge_color, style, alpha, arrowstyle, arrowsize, edge_cmap, edge_vmin, edge_vmax, ax, arrows, label, node_size, nodelist, node_shape, connectionstyle, min_source_margin, min_target_margin)\u001b[0m\n\u001b[1;32m    638\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    639\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0medgelist\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 640\u001b[0;31m         \u001b[0medgelist\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mlist\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mG\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0medges\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    641\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    642\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0medgelist\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m:\u001b[0m  \u001b[0;31m# no edges!\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mAttributeError\u001b[0m: 'set' object has no attribute 'edges'"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "nx.draw(Gmax)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
