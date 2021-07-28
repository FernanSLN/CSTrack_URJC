import utils
import keywords_icalt

hashmain = utils.get_hashtagsmain("/home/fernan/Documents/Lynguo_def2.csv", keywords_icalt.k, keywords_icalt.k_stop)

edges = utils.get_edgesMain(hashmain)

# Con las stopwords eliminamos el bot:

sortedNumberHashtags, sortedHashtagsmain = utils.prepare_hashtagsmain(edges, stopwords=['airpollution', 'luftdaten',
                                                                                        'fijnstof', 'waalre', 'pm2',
                                                                                        'pm10'])

utils.plotbarchart(10, sortedHashtagsmain, sortedNumberHashtags, 'Top 10 hashtags más utilizados',
                   'Hashtag', 'Nº de veces')