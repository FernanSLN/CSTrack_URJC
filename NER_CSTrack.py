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
descriptions = df.description.tolist()
names_short = df['name'][:3].tolist()


def preprocess(sentences):
    tokenized_list = []
    for sent in sentences:
        token = word_tokenize(str(sent))
        token = pos_tag(token)
        tokenized_list.append(token)
    return tokenized_list

names_tokenized = preprocess(names_short)


pattern = 'NP: {<DT>?<JJ>*<NN>}'

cp = nltk.RegexpParser(pattern)

ne_tree = nltk.ne_chunk(pos_tag(names_list))



# Prueba con spacey

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

doc = []
orgs = []
for item in names_list:
    process = nlp(item)
    doc.append(process)
    ner = [X.label_ for X in process.ents]
    orgs.append(ner)
    #orgs = [x for x in orgs if x != []

doc2 = []
ner_desc = []
for item in descriptions:
    process = nlp(item)
    doc2.append(process)
    ner = [(X.text, X.label_) for X in process.ents]
    ner_desc.append(ner)





df2 = pd.DataFrame({'Follower': names_list, 'Description': descriptions, 'Entity': orgs, 'ER Descriptions':ner_desc})
print(df2)


