import sys
sys.path.insert(1, '/home/fernan/Documents/Proyectos/CSTrack-URJC')
import utils
from modin_Dataframe import df
from sdgs_list import sdgs_keywords

listHashtagsRT2 = utils.get_hashtagsRT(df, keywords=sdgs_keywords)
edges = utils.get_edgesHashRT(listHashtagsRT2)
sortedHashtagsRT,sortedNumberHashtags = utils.prepare_hashtags(edges, stopwords='sdgs')

utils.plotbarchart(10, sortedNumberHashtags, sortedHashtagsRT, title=None, xlabel=None, ylabel=None)