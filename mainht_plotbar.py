import utils
import keywords_icalt

hashmain = utils.get_hashtagsmain("/home/fernando/Documentos/Lynguo_def2.csv", keywords_icalt.k, keywords_icalt.k_stop)
hashmain2, arsitasmain= utils.mainHashtags2(hashmain)

sortedNumberHashtags, sortedHashtagsmain = utils.prepare_hashtagsmain(hashmain2)

utils.plotbarchart(10, sortedHashtagsmain, sortedNumberHashtags, 'Top 10 hashtags más utilizados',
          'Hashtag', 'Nº de veces', 'graficoHashtagsUsados')