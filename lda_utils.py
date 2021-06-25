# Importing modules
import pandas as pd
import os
import utils

def format_topics_sentences(ldamodel=None, corpus=None, texts=None):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        #print(i, row_list)
        row = row_list[0] if ldamodel.per_word_topics else row_list
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            #print("j", topic_num, prop_topic)
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

def read_keywords(filename):
    with open(filename, "r", encoding="utf-8")  as f:
        text = f.read()
        return text.split("\n")

# Read data into papers
papers = pd.read_csv("/home/fernan/Documents/Lynguo_22June.csv", sep=';', encoding='latin-1', error_bad_lines=False)
#papers = utils.filter_by_topic(papers, keywords=read_keywords("sdg_keys.txt"), stopwords=None)
# Print head
print(papers.head())
print("Num filas pre format", len(papers.index))

# Remove the columns
papers = papers[['Texto', 'Usuario']].copy()
papers = papers.drop_duplicates()
papers = papers.dropna()
print("Num filas", len(papers.index))
papers = papers[['Texto']].copy().sample(len(papers.index))

# Print out the first rows of papers
#print(papers.head())

# Load the regular expression library
import re
# Remove punctuation
papers['paper_text_processed'] = papers['Texto'].map(lambda x: re.sub('[,\.!?#]', '', x))
# Convert the titles to lowercase
papers['paper_text_processed'] = papers['paper_text_processed'].map(lambda x: x.lower())
print("PRE SCI")
papers['paper_text_processed'].map(lambda x: re.sub('citizenscience', '', x))
papers['paper_text_processed'] = papers['paper_text_processed'].str.replace('citizenscience','')
print("POST SCI")

# Print out the first rows of papers
#print(papers['paper_text_processed'].head())

from wordcloud import WordCloud
# Join the different processed titles together.
long_string = ','.join(list(papers['paper_text_processed'].values))
print("Hay citizenscience", long_string.find("citizenscience"))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_file("./wc_1.png")

import gensim
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', "https"])
stop_words.extend(gensim.parsing.preprocessing.STOPWORDS)

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words and len(word) > 3] for doc in texts]

data = papers.paper_text_processed.values.tolist()
data_words = list(sent_to_words(data))
# remove stop words
data_words = remove_stopwords(data_words)

import gensim.corpora as corpora
# Create Dictionary
id2word = corpora.Dictionary(data_words)
# Create Corpus
texts = data_words
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View

from pprint import pprint
# number of topics
num_topics = 17
# Build LDA model
lda_model = gensim.models.LdaModel(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics)
# Print the Keyword in the 10 topics
#pprint(lda_model.print_topics())
print("LLEGA 1")
doc_lda = lda_model[corpus]

import pyLDAvis.gensim_models
import pickle
import pyLDAvis
# Visualize the topics
LDAvis_data_filepath = os.path.join('./ldavis_prepared_'+str(num_topics))
# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself
if 1 == 1:
    LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, n_jobs=1)

    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)
pyLDAvis.save_html(LDAvis_prepared, './ldavis_prepared_'+ str(num_topics) +'.html')
pyLDAvis.display(LDAvis_prepared)
print("LLEGA 2")
print("TOPICS 1")
df_topics = format_topics_sentences(lda_model, corpus, data)
df_topics = df_topics.reset_index()
print("TOPICS 2")
df_topics.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_topics.to_csv("dominant_topics.csv")
print("TOPICS 3")

results = df_topics.groupby("Dominant_Topic")["Dominant_Topic"].count().reset_index(name="count")
print("TOPICS 4")
print(results)

import plotly.express as px
fig = px.bar(results, x='Dominant_Topic', y='count')
fig.show()