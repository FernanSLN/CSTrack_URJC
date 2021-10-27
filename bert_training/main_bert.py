import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from tensorflow import keras
import pickle
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
    
"""df = pd.read_json('./tweets_v4.json')
df.category = df["category"].map(lambda x: 0 if x == "ODS1" or x == "goal1" or x == "SDG1" else x)
df.category = df.category.map(lambda x: 1 if x == "ODS2" or x == "goal2" or x == "SDG2" else x)
df.category = df.category.map(lambda x: 2 if x == "ODS3" or x == "goal3" or x == "SDG3" else x)
df.category = df.category.map(lambda x: 3 if x == "ODS4" or x == "goal4" or x == "SDG4" else x)
df.category = df.category.map(lambda x: 4 if x == "ODS5" or x == "goal5" or x == "SDG5" else x)
df.category = df.category.map(lambda x: 5 if x == "ODS6" or x == "goal6" or x == "SDG6" else x)
df.category= df.category.map(lambda x: 6 if x == "ODS7" or x == "goal7" or x == "SDG7" else x)
df.category = df.category.map(lambda x: 7 if x == "ODS8" or x == "goal8" or x == "SDG8" else x)
df.category = df.category.map(lambda x: 8 if x == "ODS9" or x == "goal9" or x == "SDG9" else x)
df.category = df.category.map(lambda x: 9 if x == "ODS10" or x == "goal10" or x == "SDG10" else x)
df.category = df.category.map(lambda x: 10 if x == "ODS11" or x == "goal11" or x == "SDG11" else x)
df.category = df.category.map(lambda x: 11 if x == "ODS12" or x == "goal12" or x == "SDG12" else x)
df.category = df.category.map(lambda x: 12 if x == "ODS13" or x == "goal13" or x == "SDG13" else x)
df.category = df.category.map(lambda x: 13 if x == "ODS14" or x == "goal14" or x == "SDG14" else x)
df.category = df.category.map(lambda x: 14 if x == "ODS15" or x == "goal15" or x == "SDG15" else x)
df.category = df.category.map(lambda x: 15 if x == "ODS16" or x == "goal16" or x == "SDG16" else x)
df.category = df.category.map(lambda x: 16 if x == "ODS17" or x == "goal17" or x == "SDG17" else x)"""
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
"""data = df.loc[:, ~df.columns.str.contains('Unnamed: 0', case=False)] 
data = data.loc[:, ~data.columns.str.contains('Unnamed: 0.1', case=False)] 
data = data.loc[:, ~data.columns.str.contains('Unnamed: 0.1.1', case=False)] 
data = data.loc[:, ~data.columns.str.contains('t_id', case=False)] 
data=data.rename(columns = {'headline': 'text'}, inplace = False)"""
df_osdg = pd.read_csv("osdg.csv")
df_osdg = df_osdg[["text", "sdg"]].copy()
df_osdg['sdg'] -= 1
print(df_osdg.head())
"""sentences=data['text'].values.tolist() + df_osdg["text"].values.tolist()
labels=data['category'].values.tolist() + df_osdg["sdg"].values.tolist()"""
sentences=df_osdg["text"].values.tolist()
labels=df_osdg["sdg"].values.tolist()
print("--------- labels ------------- ")
print(labels)

input_ids=[]
attention_masks=[]

num_classes=len(df_osdg.sdg.unique())

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased',num_labels=num_classes)

for sent in sentences:
    bert_inp=bert_tokenizer.encode_plus(sent,add_special_tokens = True,max_length =64,pad_to_max_length = True,return_attention_mask = True)
    input_ids.append(bert_inp['input_ids'])
    attention_masks.append(bert_inp['attention_mask'])

input_ids=np.asarray(input_ids)
attention_masks=np.array(attention_masks)
labels=np.array(labels)

print('Preparing the pickle file.....')

pickle_inp_path='./data/bert_inp.pkl'
pickle_mask_path='./data/bert_mask.pkl'
pickle_label_path='./data/bert_label.pkl'

pickle.dump((input_ids),open(pickle_inp_path,'wb'))
pickle.dump((attention_masks),open(pickle_mask_path,'wb'))
pickle.dump((labels),open(pickle_label_path,'wb'))


print('Pickle files saved as ',pickle_inp_path,pickle_mask_path,pickle_label_path)
print('Loading the saved pickle files..')

input_ids=pickle.load(open(pickle_inp_path, 'rb'))
attention_masks=pickle.load(open(pickle_mask_path, 'rb'))
labels=pickle.load(open(pickle_label_path, 'rb'))

print('Input shape {} Attention mask shape {} Input label shape {}'.format(input_ids.shape,attention_masks.shape,labels.shape))

train_inp,val_inp,train_label,val_label,train_mask,val_mask=train_test_split(input_ids,labels,attention_masks,test_size=0.2)

print('Train inp shape {} Val input shape {}\nTrain label shape {} Val label shape {}\nTrain attention mask shape {} Val attention mask shape {}'.format(train_inp.shape,val_inp.shape,train_label.shape,val_label.shape,train_mask.shape,val_mask.shape))

log_dir='tensorboard_data/tb_bert'
model_save_path='./models/newbert_model/'

callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,monitor='val_loss',mode='min',save_best_only=True),keras.callbacks.TensorBoard(log_dir=log_dir)]

print('\nBert Model',bert_model.summary())

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5,epsilon=1e-08)

bert_model.compile(loss=loss,optimizer=optimizer,metrics=[metric])
history=bert_model.fit([train_inp,train_mask],train_label,batch_size=32,epochs=4,validation_data=([val_inp,val_mask],val_label),callbacks=callbacks)

try:
	print("Modelo guardado")
	bert_model.save_weights("full_sdgs_osdg.h5")
except Exception as e:
	print(e)


