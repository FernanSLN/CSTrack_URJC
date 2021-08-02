import sys
sys.path.insert(2, '../CSTrack-URJC')
from modin_Dataframe import df
from utils.utils import tfidf_wordcloud

tfidf_wordcloud(df)
