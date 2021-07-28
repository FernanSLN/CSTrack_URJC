import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
import keras
from tqdm import tqdm
import pickle
from keras.models import Model
import keras.backend as K
from sklearn.metrics import confusion_matrix,f1_score,classification_report
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import itertools
from keras.models import load_model
from sklearn.utils import shuffle
from transformers import *
from transformers import BertTokenizer, TFBertModel, BertConfig


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def clean_stopwords_shortwords(w):
    stopwords_list=stopwords.words('english')
    words = w.split() 
    clean_words = [word for word in words if (word not in stopwords_list) and len(word) > 2]
    return " ".join(clean_words) 

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w=clean_stopwords_shortwords(w)
    w=re.sub(r'@\w+', '',w)
    return w
    
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
"""df.category = df.category.map(lambda x: 1 if x == "ODS1" or x == "goal1" or x =="SDG1" else x)
df.category = df.category.map(lambda x: "SDG2" if x == "ODS2" or x == "goal2" else x)
df.category = df.category.map(lambda x: "SDG3" if x == "ODS3" or x == "goal3" else x)
df.category = df.category.map(lambda x: "SDG4" if x == "ODS4" or x == "goal4" else x)
df.category = df.category.map(lambda x: "SDG5" if x == "ODS5" or x == "goal5" else x)
df.category = df.category.map(lambda x: "SDG6" if x == "ODS6" or x == "goal6" else x)
df.category = df.category.map(lambda x: "SDG7" if x == "ODS7" or x == "goal7" else x)
df.category = df.category.map(lambda x: "SDG8" if x == "ODS8" or x == "goal8" else x)
df.category = df.category.map(lambda x: "SDG9" if x == "ODS9" or x == "goal9" else x)
df.category = df.category.map(lambda x: "SDG10" if x == "ODS10" or x == "goal10" else x)
df.category = df.category.map(lambda x: "SDG11" if x == "ODS11" or x == "goal11" else x)
df.category = df.category.map(lambda x: "SDG12" if x == "ODS12" or x == "goal12" else x)
df.category = df.category.map(lambda x: "SDG13" if x == "ODS13" or x == "goal13" else x)
df.category = df.category.map(lambda x: "SDG14" if x == "ODS14" or x == "goal14" else x)
df.category = df.category.map(lambda x: "SDG15" if x == "ODS15" or x == "goal15" else x)
df.category = df.category.map(lambda x: "SDG16" if x == "ODS16" or x == "goal16" else x)
df.category = df.category.map(lambda x: "SDG17" if x == "ODS17" or x == "goal17" else x)"""
data = df.loc[:, ~df.columns.str.contains('Unnamed: 0', case=False)] 
data = data.loc[:, ~data.columns.str.contains('Unnamed: 0.1', case=False)] 
data = data.loc[:, ~data.columns.str.contains('Unnamed: 0.1.1', case=False)] 
data = data.loc[:, ~data.columns.str.contains('t_id', case=False)] 
data=data.rename(columns = {'headline': 'text'}, inplace = False)
sentences=data['text']
labels=data['category']
print(data.head())
input_ids=[]
attention_masks=[]

num_classes=len(data.category.unique()) + 1

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=num_classes)

for sent in sentences:
    bert_inp=bert_tokenizer.encode_plus(sent,add_special_tokens = True,max_length =64,pad_to_max_length = True,return_attention_mask = True)
    input_ids.append(bert_inp['input_ids'])
    attention_masks.append(bert_inp['attention_mask'])

input_ids=np.asarray(input_ids)
attention_masks=np.array(attention_masks)
labels=np.array(labels)

print('Preparing the pickle file.....')

train_inp,val_inp,train_label,val_label,train_mask,val_mask=train_test_split(input_ids,labels,attention_masks,test_size=0.2)


model_save_path='./models/bert_model_v3.h5'
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5,epsilon=1e-08)

trained_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=18)
trained_model.compile(loss=loss,optimizer=optimizer, metrics=[metric])
trained_model.load_weights(model_save_path)


preds = trained_model.predict([val_inp,val_mask],batch_size=32)
pred_labels = preds.logits.argmax(axis=1)
#print(val_label)
#print(pred_labels)
cm = confusion_matrix(val_label, pred_labels)
cm = np.round(cm / cm.astype(np.float).sum(axis=1),2)
print(cm)
import seaborn as sn
import matplotlib.pyplot as plt
labels = df["category"].unique()
labels = sorted(labels)
df_cm = pd.DataFrame(cm, index=labels, columns=labels)
print(df_cm)
print(labels)
plt.figure(figsize=(10,7))
sn.heatmap(df_cm, annot=True)
plt.show()

f1 = f1_score(val_label,pred_labels, average="weighted")
print('F1 score',f1)
print('Classification Report')
print(classification_report(val_label,pred_labels))
