import sys
sys.path.append('/home/fernan/Documents/Proyectos/CSTrack-URJC')
import utils
from DataFrame import df
from sdgs_list import sdgs_keywords
from webweb import Web

G = utils.kcore_Graph(df, keywords=sdgs_keywords)
web = Web(title="retweets", nx_G=G)
web.display.gravity = 1

# show the visualization
web.show()