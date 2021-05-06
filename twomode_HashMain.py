import utils
import networkx as nx
from webweb import Web

G = utils.get_twomodeHashMain('/home/fernan/Documents/Lynguo_April21.csv', keywords='SDGS')

# Graficado con Web

web = Web(title="main graph", nx_G=G)
web.display.gravity = 1


web.show()
