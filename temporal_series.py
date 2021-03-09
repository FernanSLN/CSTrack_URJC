from utils import Maindf, getDays, get_hashtagsmain, get_edgesMain, plottemporalserie, prepare_hashtagsmain

main_df = Maindf('/home/fernan/Documents/Lynguo_def2.csv')
dias = getDays(main_df)
listHt = get_hashtagsmain('/home/fernan/Documents/Lynguo_def2.csv')
edges = get_edgesMain(listHt)
sortedNH, sortedMH = prepare_hashtagsmain(edges)
plottemporalserie(dias, main_df, sortedMH, 'Evolución temporal de los hashtags más utilizados')