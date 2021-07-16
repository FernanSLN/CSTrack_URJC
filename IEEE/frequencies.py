import sys
sys.path.insert(2, '/home/fernan/Documents/Proyectos/CSTrack-URJC')
from modin_Dataframe import df
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import nltk
from nltk. tag import pos_tag
from nltk.corpus import universal_treebanks
from nltk import word_tokenize
from utils import filter_by_topic, filter_by_subtopic, filter_by_interest, tfidf_wordcloud
from sdgs_list import sdgs_keywords
import re
import keywords_icalt

df = filter_by_topic(df, keywords=keywords_icalt.k, stopwords=keywords_icalt.k_stop)
df_Text = df['Texto']
df_Text = df_Text.dropna()
df_Text = df_Text.drop_duplicates()
tvec = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
tvec_freq = tvec.fit_transform(df_Text.dropna())
freqs = np.asarray(tvec_freq.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': tvec.get_feature_names(), 'freqs': freqs})
weights_df = weights_df.sort_values(by='freqs', ascending=False)
terms_list = list(weights_df['term'])
terms_df = pd.DataFrame({'terms': terms_list})
terms_df['terms'] = terms_df['terms'].apply(nltk.word_tokenize)

unique_string = (' ').join(terms_list)
token = word_tokenize(unique_string)
token = pos_tag(token, tagset='universal', lang='eng')
nouns = [t[0] for t in token if (t[1] == 'NOUN')]

words = []
for word in nouns:
    match = re.findall("\A[a-z-A-Z]+", word)
    for object in match:
        words.append(word)

regex = re.compile(r'htt(\w+)')

words = [word for word in words if not regex.match(word)]

weights_df = weights_df[weights_df['term'].isin(words)]


freqs = list(weights_df['freqs'])
names = list(weights_df['term'])


tagged = nltk.pos_tag(names)

inverted_freqs = list(abs(np.log(freqs)))

d = {}
for key in names:
    for value in inverted_freqs:
        d[key] = value
        inverted_freqs.remove(value)
        break

unique_string2 = (' ').join(names)

wordcloud = WordCloud(width=900, height=900, background_color='white', min_font_size=10, max_words=400,
                          collocations=False, colormap='Pastel2')
wordcloud.generate_from_frequencies(frequencies=d)

plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()