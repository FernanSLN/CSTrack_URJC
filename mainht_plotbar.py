import utils
import keywords_icalt

hashmain = utils.get_hashtagsmain("/home/fernan/Documents/Lynguo_def2.csv", keywords_icalt.k, keywords_icalt.k_stop)

edges = utils.get_edgesMain(hashmain)

sortedNumberHashtags, sortedHashtagsmain = utils.prepare_hashtagsmain(edges)

utils.plotbarchart(10, sortedHashtagsmain, sortedNumberHashtags, 'Top 10 hashtags más utilizados',
                   'Hashtag', 'Nº de veces', '/home/fernan/Documents/Proyectos/CSTrack-URJC',
                   'graficoHashtagsUsados')
