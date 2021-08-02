import numpy as np
import pandas as pd
from copy import deepcopy
from bertopic import BERTopic
import re
from nltk.corpus import stopwords


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
	 	
   
def filter_by_topic(df, keywords, stopwords):
    if keywords:
        df = df[df['Texto'].str.contains("|".join(keywords), case=False).any(level=0)]
        if stopwords:
            df = df[~df['Texto'].str.contains("|".join(stopwords), case=False).any(level=0)]
    return df
    
df = pd.read_csv("tweets_june2021.csv", sep=";", encoding="latin-1", error_bad_lines=False)
with open("sdg_keys.txt", encoding="utf-8") as f:
	list_keys = f.read().split("\n")

import string
print("Documentos:", len(df["Texto"].values.tolist()))
df = filter_by_topic(df, keywords=list_keys, stopwords=None)
df["Texto"] = df["Texto"].apply(str)
df["Texto"] = df["Texto"].map(lambda x: re.sub("[,\.!?#]^¿¡","",x))
df["Texto"] = df["Texto"].map(lambda x: x.lower())
df["Texto"] = df["Texto"].map(lambda x: re.sub("climate change","climatechange",x))
stop_words = stopwords.words("english")
stop_words.extend(stopwords.words("spanish"))
stop_words.extend(["from", "citizenscience", "citizen science", "citizen", "science", "ciencia", "need", "thank", "project", "projects"])
documents = df["Texto"].values.tolist()
documents = remove_stopwords(documents, stop_words)
print("Documentos:", len(documents))
print(documents)


model = BERTopic(language="english")
topics, probs = model.fit_transform(documents)
print(model.get_topic_freq())
print(model.get_topic(0))

new_t, new_p = model.reduce_topics(documents, topics, probs, nr_topics = 17)
print(model.get_topic_freq())
fig = model.visualize_topics()
fig.show()



