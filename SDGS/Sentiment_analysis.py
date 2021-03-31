import sys
sys.path.insert(1, '/home/fernan/Documents/Proyectos/CSTrack-URJC')
import utils
import pandas as pd
import statistics as stats
import re
import numpy as np

df_sentiment = pd.read_csv('/home/fernan/Documents/Proyectos/CSTrack-URJC/SDGS/vaderSentiment.csv', sep=';',
                           encoding='utf-8', error_bad_lines=False)

df_sentiment = df_sentiment[['Texto', 'compound']]

idx = df_sentiment['Texto'].str.contains('RT @', na=False)

df_sentimentRT = df_sentiment[idx]

retweets = [list(x) for x in df_sentimentRT.to_numpy()]

arrobas = []

for row in retweets:
    reg = re.search('@(\w+)', row[0])
    if reg:
        matchRT = reg.group(1)
        row[0] = matchRT
        arrobas.append(row)

df_arrobas = pd.DataFrame(arrobas, columns=['User', 'Values'])

df_arrobas = df_arrobas.groupby('User', as_index=False).mean()

df_arrobas = df_arrobas.sort_values('Values', ascending=False)

users = df_arrobas['User']

values = df_arrobas['Values']

utils.plotbarchart(10, users, values, 'Top 10 Users with higher Sentiment', 'User', 'Sentiment')

df_SRT = df_sentimentRT.sort_values('compound', ascending=False)

dfSRT = df_SRT.drop_duplicates(subset='Texto', keep='first')

df_SRT[:50].to_csv('top50 retweets by Sentiment.csv', sep=';', index=False, decimal='.', encoding='utf-8')
