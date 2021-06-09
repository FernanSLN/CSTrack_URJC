import sys
sys.path.insert(1, '/home/fernan/Documents/Proyectos/CSTrack-URJC')
from utils import filter_by_topic, dataframe_statistics
from DataFrame import df
from sdgs_list import sdgs_keywords

print(len(df))
df_filtered = filter_by_topic(df, keywords=sdgs_keywords, stopwords=None)
print(len(df_filtered))

df_statistics = dataframe_statistics(df_filtered)
print(df_statistics)
