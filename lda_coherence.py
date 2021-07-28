# Importing modules
import pandas as pd
import os
import utils

def main():
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
    papers = pd.read_csv("topic_modelling.csv", sep=';', encoding='latin-1', error_bad_lines=False)
    papers = utils.filter_by_topic(papers, keywords=read_keywords("sdg_keys.txt"), stopwords=None)
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
    """papers['paper_text_processed'].map(lambda x: re.sub('citizenscience', '', x))
    papers['paper_text_processed'] = papers['paper_text_processed'].str.replace('citizenscience','')
    papers['paper_text_processed'].map(lambda x: re.sub(' citizen ', '', x))
    papers['paper_text_processed'] = papers['paper_text_processed'].str.replace(' science ','')
    papers['paper_text_processed'].map(lambda x: re.sub('sdgs', '', x))
    papers['paper_text_processed'] = papers['paper_text_processed'].str.replace('sdgs','')"""
    papers['paper_text_processed'].map(lambda x: re.sub('climage change', ' climatechange', x))
    papers['paper_text_processed'] = papers['paper_text_processed'].str.replace('climate change','climatechange')
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
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', "https", "citizenscience", "citizen", "science", "sdgs", "citsci", "need", "thank", "ciencia", "ciudadana", "project", "projects"])
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
    # Alpha parameter
    import numpy as np
    alpha = list(np.arange(0.01, 1, 0.3))
    alpha.append('symmetric')
    alpha.append('asymmetric')
    # Beta parameter
    beta = list(np.arange(0.01, 1, 0.3))
    beta.append('symmetric')
    texts = [[id2word[word_id] for word_id, freq in doc] for doc in corpus]
    alphas = []
    betas = []
    coherences = []
    topics = []
    for k in range(5,30):
    # Build LDA model
        lda_model = gensim.models.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=k, passes=40, iterations=200)
        cm = gensim.models.coherencemodel.CoherenceModel(model=lda_model, dictionary=id2word, texts=texts, coherence="c_v")
        coh = cm.get_coherence()
        print("La coherencia para", k, "topics es", coh)
        topics.append(k)
        coherences.append(coh)
    df = pd.DataFrame(list(zip(topics, coherences)), columns=["ntopics", "coherence"])
    df.to_csv("topic_coherence.csv")

if __name__ == "__main__":
    main()