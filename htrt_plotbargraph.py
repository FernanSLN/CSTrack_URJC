import utils
import keywords_icalt


listHashtagsRT2 = utils.get_hashtagsRT("/home/fernan/Documents/Lynguo_def2.csv", keywords_icalt.k, keywords_icalt.k_stop)
edges = utils.get_edgesHashRT(listHashtagsRT2)
sortedHashtagsRT,sortedNumberHashtags = utils.prepare_hashtags(edges)

utils.plotbarchart(10, sortedNumberHashtags, sortedHashtagsRT, 'Top 10 hashtag con más retweets',
             'Hashtag', 'Nº de veces', '/home/fernan/Documents', 'graficoHashtagsRT')
