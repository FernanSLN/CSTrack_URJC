import sys
sys.path.insert(1, '/home/fernan/Documents/Proyectos/CSTrack-URJC')
import utils
from DataFrame import df
from sdgs_list import sdgs_keywords

df_RT, dias = utils.main_or_RT_days(df, RT=True)
listHRT = utils.get_hashtagsRT(df, keywords=sdgs_keywords)
edges = utils.get_edgesHashRT(listHRT)
sortedNHRT, sortedHT = utils.prepare_hashtags(edges)
utils.plottemporalserie(dias, df_RT, sortedHT, 'Temporal evolution of top 10 retweeted hashtags', x=0, y=10)