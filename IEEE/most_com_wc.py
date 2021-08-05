from utils import filter_by_topic, plotbarchart, most_common, most_commonwc
from IEEE.modin_Dataframe import df
from IEEE.sdgs_list import sdgs_keywords

df = filter_by_topic(df, keywords=sdgs_keywords)

tuples_dict, words, numbers = most_common(df)

plotbarchart(10, words, numbers)

most_commonwc(df)
