import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import nltk
from nltk.tag import pos_tag
from nltk import word_tokenize
from utils import filter_by_topic, filter_by_subtopic, filter_by_interest

df = pd.read_csv('/home/fernan/Documents/Lynguo_April21.csv', sep=';', encoding='utf-8', error_bad_lines=False)

def tfidf_wordcloud(df, keywords=None, stopwords=None, keywords2=None, stopwords2=None, interest=None):
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    df_Text = df['Texto']
    df_Text = df_Text.dropna()
    df_Text = df_Text.drop_duplicates()
    tvec = TfidfVectorizer(min_df=0.01, max_df=0.5, stop_words='english', ngram_range=(1, 1))
    tvec_freq = tvec.fit_transform(df_Text.dropna())
    freqs = np.asarray(tvec_freq.mean(axis=0)).ravel().tolist()
    weights_df = pd.DataFrame({'term': tvec.get_feature_names(), 'freqs': freqs})
    weights_df = weights_df.sort_values(by='freqs', ascending=False)
    terms_list = list(weights_df['term'])
    terms_df = pd.DataFrame({'terms': terms_list})
    terms_df['terms'] = terms_df['terms'].apply(nltk.word_tokenize)

    nouns = []
    for index, row in terms_df.iterrows():
        nouns.extend(
            [word for word, pos in pos_tag(row[0]) if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')])

    weights_df = weights_df[weights_df['term'].isin(nouns)]

    freqs = list(weights_df['freqs'])
    names = list(weights_df['term'])
    inverted_freqs = list(abs(np.log(freqs)))

    d = {}
    for key in names:
        for value in inverted_freqs:
            d[key] = value
            inverted_freqs.remove(value)
            break

    unique_string = (' ').join(names)

    wordcloud = WordCloud(width=900, height=900, background_color='azure', min_font_size=10, max_words=400,
                          collocations=False, colormap='tab20c')
    wordcloud.generate_from_frequencies(frequencies=d)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


tfidf_wordcloud(df)