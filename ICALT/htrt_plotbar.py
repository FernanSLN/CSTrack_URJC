import utils
from ICALT.keywords_icalt import k, k_stop

listHashtagsRT2 = utils.get_hashtagsRT("/home/fernan/Documents/Lynguo_def2.csv", k, k_stop)
edges = utils.get_edgesHashRT(listHashtagsRT2)
sortedHashtagsRT,sortedNumberHashtags = utils.prepare_hashtags(edges)

utils.plotbarchart(10, sortedNumberHashtags, sortedHashtagsRT, 'Top 10 more retweeted hashtags',
             'Hashtag', 'n times')