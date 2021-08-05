import utils
from covid_keywords import covid

utils.wordcloudRT('/home/fernan/Documents/Lynguo_def2.csv', keywords=covid)
utils.wordcloudRT_logo(filename='/home/fernan/Documents/Lynguo_def2.csv', keywords=covid,
                       image='/home/fernan/Pictures/lupa.png')