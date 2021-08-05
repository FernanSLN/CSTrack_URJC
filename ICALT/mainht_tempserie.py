from utils import main_or_RT_days, get_hashtagsmain, get_edgesMain, plottemporalserie, prepare_hashtagsmain, botwords
from ICALT.keywords_icalt import k, k_stop

df_main, dias = main_or_RT_days('/home/fernan/Documents/Lynguo_April21.csv', RT=False)
listHt = get_hashtagsmain('/home/fernan/Documents/Lynguo_April21.csv', keywords=k, stopwords=k_stop)
edges = get_edgesMain(listHt)
sortedNH, sortedMainH = prepare_hashtagsmain(edges, stopwords=botwords)
plottemporalserie(dias, df_main, sortedMainH, 'Temporal evolution of the used hashtags', x=0, y=6)