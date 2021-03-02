import sys
sys.path.insert(1, '/home/fernan/Documents/Proyectos/CSTrack-URJC')
import utils
import RTcovid_graph

utils.wordcloudmain('/home/fernan/Documents/Lynguo_def2.csv',keywords=RTcovid_graph.covid)
utils.wordcloud_mainhtlogo('/home/fernan/Documents/Lynguo_def2.csv', image='/home/fernan/Pictures/lupa.png',
                           keywords=RTcovid_graph.covid)

