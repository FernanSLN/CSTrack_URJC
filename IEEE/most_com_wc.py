import sys
sys.path.insert(1, '/home/fernan/Documents/Proyectos/CSTrack-URJC')
from utils import utils
from modin_Dataframe import df
from collections import Counter
import re
from nltk import pos_tag
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import string
import matplotlib.pyplot as plt
import keywords_icalt

df = utils.filter_by_topic(df, keywords=keywords_icalt.k, stopwords=keywords_icalt.k_stop)

#most_common_list = utils.most_common(df, number=50)

#utils.most_commonwc(df)


subset = df['Texto']
subset = subset.dropna()
words = " ".join(subset).lower().split()
token = pos_tag(words, tagset='universal', lang='eng')
word_list = [t[0] for t in token if (t[1] == 'NOUN', t[1] == 'VERB', t[1] == 'ADJ')]
hashtags = []
users = []
interrogations = []

print('Llega 1')

words = []
for word in word_list:
    match = re.findall("\A[a-z-A-Z]+", word)
    for object in match:
        words.append(word)

regex = re.compile(r'htt(\w+)')

words = [word for word in words if not regex.match(word)]

print('Llega 2')
count_word = Counter(words)

#The alternative option elimante words with hashtags and userrs is:
#
print('Llega 3')

s = stopwords.words('english')
e = stopwords.words('spanish')
r = STOPWORDS
d = stopwords.words('german')
p = string.punctuation
new_elements = ('\\n', 'rt', '?', '¿', '&', 'that?s', '??', '-', 'the', 'to', 'co', 'n', 'https', 'we?re', 'everyone?s',
                'supporters?', 'z', 'here:', 'science,', 'project.', 'citizen', 'science', 'us', 'student?', 'centre?', 'science?',
                ')', 'media?)', 'education?', 'reuse,', 'older!', 'scientists?', 'don?t', 'it?s', 'i?m', 'w/', 'w', 'more:')
s.extend(new_elements)
s.extend(e)
s.extend(r)
s.extend(d)
s.extend(p)
stopset = set(s)

print('Llega 4')

for word in stopset:
    del count_word[word]

wordcloud = WordCloud(width=900, height=900, background_color='white', stopwords=stopset,
                      min_font_size=10, max_words=300, collocations=False,
                      colormap='winter').generate_from_frequencies(count_word)
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

tuples_dict = sorted(count_word.items(), key=lambda x: x[1], reverse=True)
words_pt = []
numbers_pt = []

for tuple in tuples_dict:
    words_pt.append(tuple[0])
    numbers_pt.append(tuple[1])

plt.figure(figsize=(10, 8))
plt.bar(x=words_pt[:10], height=numbers_pt[:10], color='lightsteelblue')
plt.xlabel('words', fontsize=15)
plt.ylabel('ntimes', fontsize=15)
plt.xticks(rotation=45)
plt.title('Top 10 most used words', fontsize=20, fontweight='bold')
plt.tight_layout()
plt.show()