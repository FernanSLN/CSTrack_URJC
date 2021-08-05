import utils
import pandas as pd
from modin_Dataframe import df
from sdgs_list import sdgs_keywords
import matplotlib.pyplot as plt
import seaborn as sns

def get_all(df, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None):
    df = utils.filter_by_interest(df, interest)
    df = utils.filter_by_topic(df, keywords, stopwords)
    df = utils.filter_by_subtopic(df, keywords2, stopwords2)
    df_text = df[['Usuario', 'Texto']].copy()
    df_text = df_text.dropna()
    list_text = df_text['Texto'].to_numpy()
    return list_text

def main_or_RT_days(filename):
    df = filename
    df = df[['Fecha', 'Usuario', 'Texto']]
    subset = df.dropna()
    subset['Fecha'] = pd.to_datetime(subset['Fecha'], errors='coerce')
    subset = subset.dropna()
    subset['Fecha'] = subset['Fecha'].dt.date

    # Obtenemos los días en el subset:
    df_Fecha = subset['Fecha']
    days = pd.unique(df_Fecha)
    days.sort()

    return subset, days

list_text = get_all(df, keywords=sdgs_keywords)
edges = utils.get_edgesMain(list_text)
sortedHashtags,sortedNumberHashtags = utils.prepare_hashtags(edges, stopwords='sdgs')

#utils.plotbarchart(10, sortedNumberHashtags, sortedHashtags, 'Top 10 most used hashtags',
             #'Hashtag', 'n times')

from matplotlib.pyplot import subplots

fig, ax = subplots()

"""fig = px.bar(results, x='names', y='count')
fig.update_layout({
    "plot_bgcolor" : 'rgba(0,0,0,0)',
})"""

sns.set()
plt.figure(figsize=(10, 8))
ax.bar(sortedNumberHashtags[:10], sortedHashtags[:10], color= "lightsteelblue")
plt.xticks(rotation=45)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()