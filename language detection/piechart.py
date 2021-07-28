import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

def filter_by_topic(df, keywords, stopwords):
    if keywords:
        df = df[df['Text'].str.contains("|".join(keywords), case=False).any(level=0)]
        if stopwords:
            df = df[~df['Text'].str.contains("|".join(stopwords), case=False).any(level=0)]
    return df


df = pd.read_csv("language_prediction.csv", sep=";", encoding="utf-8")

df = df.drop_duplicates(subset="Text", keep="first")

languages = list(df["Languages"])

count = Counter(languages)

languages = list(count.keys())

ntimes = list(count.values())

print(languages)

print(ntimes)

labels = languages

patches, texts = plt.pie(ntimes, startangle=90, autopct='%1.1f%%')
plt.legend(patches, labels, loc="best")
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
plt.tight_layout()
plt.show()
