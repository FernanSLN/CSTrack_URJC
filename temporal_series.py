from utils import main_or_RT_days, get_hashtagsmain, get_edgesMain, plottemporalserie, prepare_hashtagsmain, botwords

df_main, dias = main_or_RT_days('/home/fernan/Documents/Lynguo_April21.csv', RT=False)
listHt = get_hashtagsmain('/home/fernan/Documents/Lynguo_April21.csv')
edges = get_edgesMain(listHt)
sortedNH, sortedMainH = prepare_hashtagsmain(edges, stopwords=botwords)
plottemporalserie(dias, df_main, sortedMainH, 'Evolución temporal de los hashtags más utilizados', n=7)