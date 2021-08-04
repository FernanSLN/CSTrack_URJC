import utils
from covid_keywords import covid

hashmain = utils.get_hashtagsmain("/home/fernan/Documents/Lynguo_def2.csv", keywords=covid)

edges = utils.get_edgesMain(hashmain)

sortedNumberHashtags, sortedHashtagsmain = utils.prepare_hashtagsmain(edges)

utils.plotbarchart(10, sortedHashtagsmain, sortedNumberHashtags, 'Top 10 more used hashtags',
                   'Hashtag', 'n times')
