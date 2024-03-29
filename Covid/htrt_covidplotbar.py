import utils
from covid_keywords import covid

listhtrt = utils.get_hashtagsRT('/home/fernan/Documents/Lynguo_def2.csv', keywords=covid)

edges = utils.get_edgesHashRT(listhtrt)

sortedHashtagsRT,sortedNumberHashtags = utils.prepare_hashtags(edges)

utils.plotbarchart(10, sortedNumberHashtags, sortedHashtagsRT, 'Top 10 more retweeted hashtags',
             'Hashtag', 'n times')