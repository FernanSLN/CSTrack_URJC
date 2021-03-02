import utils
import RTcovid_graph

listhtrt = utils.get_hashtagsRT('/home/fernan/Documents/Lynguo_def2.csv', keywords=RTcovid_graph.covid)

edges = utils.get_edgesHashRT(listhtrt)

sortedHashtagsRT,sortedNumberHashtags = utils.prepare_hashtags(edges)

utils.plotbarchart(10, sortedNumberHashtags, sortedHashtagsRT, 'Top 10 hashtag con más retweets',
             'Hashtag', 'Nº de veces')