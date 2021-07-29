import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

def filter_by_topic(df, keywords, stopwords):
    if keywords:
        df = df[df['Text'].str.contains("|".join(keywords), case=False).any(level=0)]
        if stopwords:
            df = df[~df['Text'].str.contains("|".join(stopwords), case=False).any(level=0)]
    return df

def plotbarchart(x, y, title, xlabel, ylabel):
    sns.set()
    plt.figure(figsize=(10, 8))
    plt.bar(x=x, height=y, color='lightsteelblue')
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.xticks(rotation=45)
    plt.title(title, fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.show()

file = open("sdg_keys.txt", "r")
lines = file.readlines()
sdgs_keywords = []

for l in lines:
    sdgs_keywords.append(l.replace("\n", ""))


df = pd.read_csv("language_prediction.csv", sep=";", encoding="utf-8")

df = df.drop_duplicates(subset="Text", keep="first")


languages = list(df["Languages"])

count = Counter(languages)

language = list(count.keys())

ntimes = list(count.values())

print(language)
print(ntimes)

plotbarchart(language, ntimes, "Language distribution in Lynguo DataFrame", "Languages", "ntimes")

