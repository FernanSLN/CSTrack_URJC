import pandas as pd
from collections import Counter
from utils import plotbarchart

# This filter by topic function is specially designed for the language_detection csv:

def filter_topic(df, keywords, stopwords):
    if keywords:
        df = df[df['Text'].str.contains("|".join(keywords), case=False).any(level=0)]
        if stopwords:
            df = df[~df['Text'].str.contains("|".join(stopwords), case=False).any(level=0)]
    return df

df = pd.read_csv("language_prediction.csv", sep=";", encoding="utf-8")

# Duplicates are dropped to eliminate those redundant tweets that will blur the final count of languages

df = df.drop_duplicates(subset="Text", keep="first")

# Again, here you can apply a filter by topic in case it was not applied before:
# df = filter_topic(df, keywords=[...], stopwords=[...])

languages = list(df["Languages"])

count = Counter(languages)

language = list(count.keys())

ntimes = list(count.values())

print(language)
print(ntimes)

plotbarchart(numberbars=len(language), x=language, y=ntimes, title="Language distribution in Lynguo DataFrame",
             xlabel="Languages", ylabel="ntimes")

