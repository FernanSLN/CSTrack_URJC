import utils
from IEEE.modin_Dataframe import df
from IEEE.sdgs_list import sdgs_keywords

df_all, dias = utils.main_or_RT_days(df, RT=False)
listHRT = utils.get_hashtagsmain(df, keywords=sdgs_keywords)
edges = utils.get_edgesMain(listHRT)
sortedNHRT, sortedHT = utils.prepare_hashtagsmain(edges, stopwords=['b', 'opendata', 'wind', 'temperature', 'summary',
                                                                    'pressure', 'precipitation', 'podsumowaniednia',
                                                                    'meteorologia', 'katowice', 'humidity',
                                                                    'davisvantagepro2', 'dane', 'sdgs', 'sdg'])

utils.plottemporalserie(dias, df_all, sortedHT, 'Temporal evolution of top 10 used hashtags \n outside retweets', x=0, y=10)