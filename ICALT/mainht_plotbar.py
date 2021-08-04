import utils
from ICALT.keywords_icalt import k, k_stop

hashmain = utils.get_hashtagsmain("/home/fernan/Documents/Lynguo_def2.csv", k, k_stop)

edges = utils.get_edgesMain(hashmain)

# stopwords to eliminate the bot:

sortedNumberHashtags, sortedHashtagsmain = utils.prepare_hashtagsmain(edges, stopwords=['airpollution', 'luftdaten',
                                                                                        'fijnstof', 'waalre', 'pm2',
                                                                                        'pm10'])

utils.plotbarchart(10, sortedHashtagsmain, sortedNumberHashtags, 'Top 10 more used hashtags',
                   'Hashtag', 'n times')