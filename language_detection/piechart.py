import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from language_detection.language_distribution import filter_topic

df = pd.read_csv("language_prediction.csv", sep=";", encoding="utf-8")

# Here, once more, if not done before the filter can be done:
# df = filter_topic(df, keyw.....)

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
