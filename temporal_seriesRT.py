from utils import dfRT, getDays, plottemporalserie, get_hashtagsRT, get_edgesHashRT, prepare_hashtags

df_RT = dfRT(filename='/home/fernan/Documents/Lynguo_def2.csv')
dias = getDays(df_RT)
listHRT = get_hashtagsRT('/home/fernan/Documents/Lynguo_def2.csv')
edges = get_edgesHashRT(listHRT)
sortedNHRT, sortedHT = prepare_hashtags(edges)
plottemporalserie(dias, df_RT, sortedHT, 'Evolución temporal de los hashtags más utilizados en los Retuits')