import sys
sys.path.insert(1, '/home/fernan/Documents/Proyectos/CSTrack-URJC')
import utils
from DataFrame import df
from sdgs_list import sdgs_keywords

listHashtagsRT2 = utils.get_hashtagsRT(df, keywords=sdgs_keywords)
edges = utils.get_edgesHashRT(listHashtagsRT2)
sortedHashtagsRT,sortedNumberHashtags = utils.prepare_hashtags(edges)

utils.plotbarchart(10, sortedNumberHashtags, sortedHashtagsRT, 'Top 10 retweeted hashtags',
             'Hashtag', 'n times')