from IEEE.modin_Dataframe import df
from IEEE.sdgs_list import sdgs_keywords
from utils import tfidf_wordcloud

tfidf_wordcloud(df, keywords=sdgs_keywords)
