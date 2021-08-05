import utils
from covid_keywords import covid

utils.wordcloudmain('/home/fernan/Documents/Lynguo_def2.csv', keywords=covid)
utils.wordcloud_mainhtlogo('/home/fernan/Documents/Lynguo_def2.csv', image='/home/fernan/Pictures/lupa.png',
                           keywords=covid)

