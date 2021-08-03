from transformers import BertTokenizer
from transformers import *
import tensorflow as tf
import pandas as pd
import numpy as np

from utils import utils as u_cs

model_save_path='./models/bert_model_v2.h5'

df = pd.read_json('./tweets.json')
df.category = df["category"].map(lambda x: 1 if x == "ODS1" or x == "goal1" or x == "SDG1" else x)
df.category = df.category.map(lambda x: 2 if x == "ODS2" or x == "goal2" or x == "SDG2" else x)
df.category = df.category.map(lambda x: 3 if x == "ODS3" or x == "goal3" or x == "SDG3" else x)
df.category = df.category.map(lambda x: 4 if x == "ODS4" or x == "goal4" or x == "SDG4" else x)
df.category = df.category.map(lambda x: 5 if x == "ODS5" or x == "goal5" or x == "SDG5" else x)
df.category = df.category.map(lambda x: 6 if x == "ODS6" or x == "goal6" or x == "SDG6" else x)
df.category= df.category.map(lambda x: 7 if x == "ODS7" or x == "goal7" or x == "SDG7" else x)
df.category = df.category.map(lambda x: 8 if x == "ODS8" or x == "goal8" or x == "SDG8" else x)
df.category = df.category.map(lambda x: 9 if x == "ODS9" or x == "goal9" or x == "SDG9" else x)
df.category = df.category.map(lambda x: 10 if x == "ODS10" or x == "goal10" or x == "SDG10" else x)
df.category = df.category.map(lambda x: 11 if x == "ODS11" or x == "goal11" or x == "SDG11" else x)
df.category = df.category.map(lambda x: 12 if x == "ODS12" or x == "goal12" or x == "SDG12" else x)
df.category = df.category.map(lambda x: 13 if x == "ODS13" or x == "goal13" or x == "SDG13" else x)
df.category = df.category.map(lambda x: 14 if x == "ODS14" or x == "goal14" or x == "SDG14" else x)
df.category = df.category.map(lambda x: 15 if x == "ODS15" or x == "goal15" or x == "SDG15" else x)
df.category = df.category.map(lambda x: 16 if x == "ODS16" or x == "goal16" or x == "SDG16" else x)
df.category = df.category.map(lambda x: 17 if x == "ODS17" or x == "goal17" or x == "SDG17" else x)
categories = df.groupby('category').size().index.tolist()
category_int = {}
int_category = {}
for i, k in enumerate(categories):
    category_int.update({k:i})
    int_category.update({i:k})
    
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5,epsilon=1e-08)

trained_model = TFBertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased',num_labels=18)
trained_model.compile(loss=loss,optimizer=optimizer, metrics=[metric])
trained_model.load_weights(model_save_path)


df_test = pd.read_csv("tweets_june2021.csv", sep=";", encoding="latin-1", error_bad_lines=False)
with open("sdg_keys.txt", encoding="utf-8") as f:
	list_keys = f.read().split("\n")
df_test = u_cs.filter_by_topic(df_test, keywords=list_keys, stopwords=None)
list_tweets = pd.concat([df_test["Texto"]], axis=0).astype("str")
sentences = []
input_ids = []
attention_masks = []
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
for t in list_tweets:
	sentences.append(t)

for sent in sentences:
    bert_inp=bert_tokenizer.encode_plus(sent,add_special_tokens = True,max_length =64,pad_to_max_length = True,return_attention_mask = True)
    input_ids.append(bert_inp['input_ids'])
    attention_masks.append(bert_inp['attention_mask'])
input_ids=np.asarray(input_ids)
attention_masks=np.array(attention_masks)
preds = trained_model.predict([input_ids,attention_masks],batch_size=32)
print(preds.logits.argmax())

pred_labels = preds.logits.argmax(axis=1)
"""f1 = f1_score(labels,pred_labels)
print('F1 score',f1)
print('Classification Report')
print(classification_report(labels,pred_labels,target_names=target_names))"""
category_list = []
for i,c in enumerate(preds.logits):
	cat_n = c.argmax() - 1
	print("DATA WE HAVE:", c.argmax())
	category_list.append("SDG"+ str(int_category[cat_n]))
	print("Category:", int_category[cat_n])
	print("DESCRIPTION", sentences[i])
	
df_test["SDG"] = category_list
df_test.to_csv("tweets_with_category_bert_v2.csv")
