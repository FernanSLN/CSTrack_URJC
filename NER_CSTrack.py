import pymongo
import pandas as pd
from pymongo import MongoClient
import pprint
import nltk
from nltk import word_tokenize, pos_tag


client = MongoClient("f-l2108-pc09.aulas.etsit.urjc.es", port=21000)

db = client['cstrack']['cstrack_followers']

docs = db.find()

df = pd.DataFrame(list(db.find()))

df = df[['name', 'description']]

df_names = df['name']
names_list = df_names.tolist()
names_short = df['name'][:3].tolist()
print(names_short)

def preprocess(sentences):
    tokenized_list = []
    for sent in sentences:
        token = word_tokenize(str(sent))
        token = pos_tag(token)
        tokenized_list.append(token)
    return tokenized_list

names_tokenized = preprocess(names_short)

print(names_tokenized)

pattern = 'NP: {<DT>?<JJ>*<NN>}'

cp = nltk.RegexpParser(pattern)

ne_tree = nltk.ne_chunk(pos_tag(names_list))
print(ne_tree)



# Prueba con spacey

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

doc = []
for item in names_list:
    process = nlp(item)
    doc.append(process)



