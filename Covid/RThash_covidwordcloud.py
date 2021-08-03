import sys
sys.path.insert(1, '/home/fernan/Documents/Proyectos/CSTrack-URJC')
from utils import utils
import RTcovid_graph

utils.wordcloudRT('/home/fernan/Documents/Lynguo_def2.csv', keywords=RTcovid_graph.covid)
utils.wordcloudRT_logo(filename='/home/fernan/Documents/Lynguo_def2.csv', keywords=RTcovid_graph.covid,
                       image='/home/fernan/Pictures/lupa.png')