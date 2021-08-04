import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
from utils import filter_by_topic

warnings.simplefilter("ignore")


data = pd.read_csv("Language Detection.csv")

df = pd.read_csv("/home/fernan/Documents/Lynguo_22July.csv", sep=";", encoding="utf-8", error_bad_lines=False)

print(data.head(10))

X = data["Text"]

y = data["Language"]

le = LabelEncoder()

y = le.fit_transform(y)

data_list = []
# iterating through all the text
for text in X:
       # removing the symbols and numbers
        text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
        text = re.sub(r'[[]]', ' ', text)
        # converting the text to lower case
        text = text.lower()
        # appending to data_list
        data_list.append(text)

print(type(data_list))

cv = CountVectorizer()

X = cv.fit_transform(data_list).toarray()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

model = MultinomialNB()

model.fit(X, y)

y_pred = model.predict(x_test)

ac = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

# We get the accuracy of the model:

print("Accuracy is :", ac)

# We print the confusion matrix:

plt.figure(figsize=(15,10))

sns.heatmap(cm, annot = True)

plt.show()

# From here we select the text from our DF, we can apply the filter by topic before if we want to check the languages of
# one smaller and filtered dataset, just remove the next # and apply the keywords:
#df = filter_by_topic(df, keywords=[...], stopwords=[...])

df_text = df["Texto"]
df_text.dropna()

# The following function treats each text individually and returns the prediction

def predict_text(df):
    languages = []
    for sentence in df:
        x = cv.transform([str(sentence)]).toarray()  # converting text to bag of words model (Vector)
        lang = model.predict(x)  # predicting the language
        lang = le.inverse_transform(lang)
        languages.append(lang)

    textos = list(df)
    data = {"Text": textos, "Languages": languages}

    df_predicted = pd.DataFrame(data)
    return df_predicted


df_predict = predict_text(df_text)

df_predict.to_csv("Language prediction", sep=";", encoding='utf-8')
