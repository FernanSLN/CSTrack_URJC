from utils import main_or_RT_days, plottemporalserie, get_hashtagsRT, get_edgesHashRT, prepare_hashtags

df_RT, dias = main_or_RT_days('/home/fernan/Documents/Lynguo_April21.csv', RT=True)
listHRT = get_hashtagsRT('/home/fernan/Documents/Lynguo_def2.csv')
edges = get_edgesHashRT(listHRT)
sortedNHRT, sortedHT = prepare_hashtags(edges)
plottemporalserie(dias, df_RT, sortedHT, 'Temporal evolution of the retweeted hashtags', x=0, y=6)