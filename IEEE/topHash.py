import sys
sys.path.insert(1, '/home/fernan/Documents/Proyectos/CSTrack-URJC')
import utils
from modin_Dataframe import df
from sdgs_list import sdgs_keywords

hashmain = utils.get_hashtagsmain(df, keywords=sdgs_keywords)

edges = utils.get_edgesMain(hashmain)


sortedNumberHashtags, sortedHashtagsmain = utils.prepare_hashtagsmain(edges, stopwords=['b', 'opendata', 'wind', 'temperature',
                                                                                        'summary', 'pressure', 'precipitation',
                                                                                        'podsumowaniednia', 'meteorologia',
                                                                                        'katowice', 'humidity', 'davisvantagepro2', 'dane', 'sdgs'])

utils.plotbarchart(10, sortedHashtagsmain, sortedNumberHashtags, 'Top 10 most used hashtags outside retweets',
                   'Hashtag', 'n times')