import utils
from IEEE.modin_Dataframe import df
from IEEE.sdgs_list import sdgs_keywords
import matplotlib.pyplot as plt
import seaborn as sns

listHashtagsRT2 = utils.get_hashtagsRT(df, keywords=sdgs_keywords)
edges = utils.get_edgesHashRT(listHashtagsRT2)
sortedHashtagsRT,sortedNumberHashtags = utils.prepare_hashtags(edges, stopwords='sdgs')

# utils.plotbarchart(10, sortedNumberHashtags, sortedHashtagsRT, title=None, xlabel=None, ylabel=None)

from matplotlib.pyplot import subplots

fig, ax = subplots()

"""fig = px.bar(results, x='names', y='count')
fig.update_layout({
    "plot_bgcolor" : 'rgba(0,0,0,0)',
})"""

sns.set()
plt.figure(figsize=(10, 8))
ax.bar(sortedNumberHashtags[:10], sortedHashtagsRT[:10], color= "lightsteelblue")
plt.xticks(rotation=45)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()