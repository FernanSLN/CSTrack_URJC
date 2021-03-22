import sys
sys.path.insert(1, '/home/fernan/Documents/Proyectos/CSTrack-URJC')
import utils
from RTcovid_graph import covid

hashmain = utils.get_hashtagsmain("/home/fernan/Documents/Lynguo_def2.csv", keywords=covid)

edges = utils.get_edgesMain(hashmain)

print(edges)

sortedNumberHashtags, sortedHashtagsmain = utils.prepare_hashtagsmain(edges)

utils.plotbarchart(10, sortedHashtagsmain, sortedNumberHashtags, 'Top 10 hashtags más utilizados',
                   'Hashtag', 'Nº de veces')
