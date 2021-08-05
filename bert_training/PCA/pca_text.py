import csv
import os
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from mlxtend.frequent_patterns import fpgrowth
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from copy import deepcopy
from bertopic import BERTopic
import re
import nltk
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
import json
import prep_text

nltk.download('wordnet')

class LemmaTokenizer(object):

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, document):
        lemmas = []

        # Pre-proccessing of one document at the time

        # Removing puntuation
        translator_1 = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        document = document.translate(translator_1)

        # Removing numbers
        document = re.sub(r'\d+', ' ', document)

        # Removing special characters
        document = re.sub(r"[^a-zA-Z0-9]+", ' ', document)

        # The document is a string up to now, after word_tokenize(document) we'll work on every word one at the time
        print(document)
        for token in word_tokenize(document):
            print(token)
            # Removing spaces
            token = token.strip()

            # Lemmatizing
            token = self.lemmatizer.lemmatize(token)

            # Removing stopwords
            if len(token) > 2:
                lemmas.append(token)
        return lemmas


def generate_wordclouds(X, in_X_tfidf, k, in_word_positions):

    # Clustering
    print(X)
    in_model = KMeans(n_clusters=k, random_state=42, n_jobs=-1)
    in_y_pred = in_model.fit_predict(X)
    in_cluster_ids = set(in_y_pred)
    silhouette_avg = silhouette_score(X, in_y_pred)
    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)

    # Number of words with highest tfidf score to display
    top_count = 100
    count = 0
    for in_cluster_id in in_cluster_ids:
        # compute the total tfidf for each term in the cluster
        in_tfidf = in_X_tfidf[in_y_pred == in_cluster_id]
        # numpy.matrix
        tfidf_sum = np.sum(in_tfidf, axis=0)
        # numpy.array of shape (1, X.shape[1])
        tfidf_sum = np.asarray(tfidf_sum).reshape(-1)
        top_indices = tfidf_sum.argsort()[-top_count:]
        term_weights = {in_word_positions[in_idx]: tfidf_sum[in_idx] for in_idx in top_indices}
        with open("./term_weights/cluster_"+str(count)+".json", "w") as f:
            f.write(json.dumps(term_weights))
            f.close()
        wc = WordCloud(width=1200, height=800, background_color="white")
        wordcloud = wc.generate_from_frequencies(term_weights)
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        fig.suptitle(f"Cluster {in_cluster_id}")
        plt.savefig("./wc_images/cluster_" + str(count) + ".png")
        count +=1
        plt.show()

    return in_cluster_ids



def remove_stopwords(texts, stop_words):
    list_texts = []
    for text in texts:
        words = text.split(" ")
        final_str = ""
        for word in words:
            if word not in stop_words and len(word) > 3:
                final_str += word + " "
        final_str = final_str.strip()
        list_texts.append(final_str)
    return list_texts


def select_n_components(var_ratio, goal_var: float) -> int:
    # Set initial variance explained so far
    total_variance = 0.0

    # Set initial number of features
    n_components = 0

    # For the explained variance of each feature:
    for explained_variance in var_ratio:

        # Add the explained variance to the total
        total_variance += explained_variance

        # Add one to the number of components
        n_components += 1

        # If we reach our goal level of explained variance
        if total_variance >= goal_var:
            # End the loop
            break

    # Return the number of components
    return n_components

df = pd.read_csv("tweets_june2021.csv", sep=";", encoding="latin-1", error_bad_lines=False)
print("Documentos pre dups:", len(df["Texto"].values.tolist()))
df = df.drop_duplicates(subset=['Texto', 'Usuario'], keep="last").reset_index(drop=True)
import string

print("Documentos:", len(df["Texto"].values.tolist()))
df["Texto"] = df["Texto"].apply(str)
df["Texto"] = df["Texto"].map(lambda x: re.sub("[,\.!?#]^¿¡", "", x))
df["Texto"] = df["Texto"].map(lambda x: x.lower())
df["Texto"] = df["Texto"].map(lambda x: re.sub("climate change", "climatechange", x))
df = df[df['Texto'].notna()]
df = prep_text.nlp(df)
docs = df["token"].values.tolist()

stop_words = stopwords.words("english")
stop_words.extend(stopwords.words("spanish"))
stop_words.extend(["nan", None, "NAN", "NaN", "from", "citizenscience", "citizen science", "citizen", "science", "ciencia", "need", "thank", "project", "projects"])
docs = remove_stopwords(docs, stop_words)
print("Documentos:", len(docs))
print(docs)

# Custom tokenizer for tfidf representation
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
#vectorizer = TfidfVectorizer(max_df=.65, min_df=1, stop_words=None, use_idf=True, norm=None, tokenizer=LemmaTokenizer())

# Here we need the correct path in order to give it to the vectorizer
print("Generating TFIDF sparse matrix...")
X_tfidf = vectorizer.fit_transform(docs)
X_sparse = csr_matrix(X_tfidf)
"""print("SHAPEEEEEEEE:" ,X_sparse.shape[1]-1)
print("X_TFIDF--------------------------------------------------------------")
print(X_tfidf)"""
svd = TruncatedSVD(n_components=2000, random_state=42)
X_svd = svd.fit_transform(X_tfidf)

tsvd_var_ratios = svd.explained_variance_ratio_
cum_variance = np.cumsum(svd.explained_variance_ratio_)
print(cum_variance)
idx = np.argmax(cum_variance > .5)
print("NUMBER OF COMPONENTS ************ ", idx)
svd = TruncatedSVD(n_components=idx, random_state=42)
X_svd = svd.fit_transform(X_tfidf)

"""print("X_SVD1--------------------------------------------------------------")
print(X_svd)"""
print(f"Total variance explained: {np.sum(svd.explained_variance_ratio_):.2f}")
word_positions = {v: k for k, v in vectorizer.vocabulary_.items()}
# The variance explained is quite low for real applications. We will investigate it later.

"""word_positions = {v: k for k, v in vectorizer.vocabulary_.items()}

model = KMeans(n_clusters=100, random_state=42, n_jobs=-1)
y_pred = model.fit_predict(X_svd)
cluster_ids = set(y_pred)

min_support = 0.3
dist_words = sorted(v for k, v in word_positions.items()) # distinct words in the vocabulary
for cluster_id in cluster_ids:
    print(f"FP-Growth results on Cluster {cluster_id} with min support {min_support}")
    tfidf = X_tfidf[y_pred == cluster_id]
    # encoded as binary "presence/absence" representation as required by mlxtend
    tfidf[tfidf > 0] = 1
    # df is a pandas sparse dataframe
    df = pd.DataFrame.sparse.from_spmatrix(tfidf, columns=dist_words)
    fset = fpgrowth(df, min_support=min_support, use_colnames=True).sort_values(by='support', ascending=False)
    print(fset, '\n')

print("X_TFIDF--------------------------------------------------------------")
print(X_tfidf)
svd = TruncatedSVD(n_components=2000, random_state=42)
X_svd = svd.fit_transform(X_tfidf)
print("X_SVD1--------------------------------------------------------------")
print(X_svd)

svd = TruncatedSVD(n_components=idx, random_state=42)
X_svd = svd.fit_transform(X_tfidf)
print("X_SVD1--------------------------------------------------------------")
print(X_svd)"""
_ = generate_wordclouds(X_svd, X_tfidf, 100, word_positions)