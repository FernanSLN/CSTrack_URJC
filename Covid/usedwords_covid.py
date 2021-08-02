import sys
sys.path.insert(1, '/home/fernan/Documents/Proyectos/CSTrack-URJC')
from utils import utils
import RTcovid_graph
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import string
from nltk.corpus import stopwords

def most_common(df,number=None):
    subset = df['Texto']
    subset = subset.dropna()
    # Definimos stopwords en varios idiomas y símbolos que queremos eliminar del resultado
    s = stopwords.words('english')
    e = stopwords.words('spanish')
    r = STOPWORDS
    d = stopwords.words('german')
    p = string.punctuation
    new_elements = ('\\n', 'rt', '?', '¿', '&', 'that?s', '??', '-', '???')
    s.extend(new_elements)
    s.extend(e)
    s.extend(r)
    s.extend(d)
    s.extend(p)
    s = set(s)
    # Calculamos la frecuencia de las palabras
    word_freq = Counter(" ".join(subset).lower().split())
    for word in s:
        del word_freq[word]
    return word_freq.most_common(number)

def most_commonwc(filename):
    subset = df['Texto']
    subset = subset.dropna()
    s = stopwords.words('english')
    e = stopwords.words('spanish')
    r = STOPWORDS
    d = stopwords.words('german')
    new_elements = ('\\n', 'rt', '?', '¿', '&', 'that?s', '??', '-','the', 'to')
    s.extend(new_elements)
    s.extend(e)
    s.extend(r)
    s.extend(d)
    stopset = set(s)
    word_freq = Counter(" ".join(subset).lower().split())
    for word in s:
        del word_freq[word]
    wordcloud = WordCloud(width=900, height=900, background_color='white', stopwords=stopset,
                          min_font_size=10, max_words=10405, collocations=False,
                          colormap='winter').generate_from_frequencies(word_freq)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

df = pd.read_csv('/home/fernan/Documents/Lynguo_def2.csv', sep=';', encoding='latin-1', error_bad_lines=False)
df_covid = utils.filter_by_topic(df, keywords=RTcovid_graph.covid, stopwords=None)

most_common = most_common(df_covid,10)

names = [item[0] for item in most_common]

numbers = [item[1] for item in most_common]

utils.plotbarchart(10, names, numbers, 'top 10 palabras mas usadas', 'palabras', 'numero de veces')


most_commonwc(df_covid)