import sys
sys.path.insert(1, '/home/fernan/Documents/Proyectos/CSTrack-URJC')
import utils
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd


analyser = SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(sentence):
        score = analyser.polarity_scores(sentence)
        return score

def sentiment_analyser(filename):
    df = pd.read_csv(filename, sep=';', encoding='utf-8', error_bad_lines=False)
    df = filter_by_interest(df, interest)
    df = filter_by_topic(df, keywords, stopwords)
    df = filter_by_subtopic(df, keywords2, stopwords2)
    df = df[['Texto', 'Usuario']]
    df = df.dropna()
    Users = df['Usuario']
    Texto = df['Texto']
    sentences = Texto
    list_of_dicts = []
    for sentence in sentences:
        adict = analyser.polarity_scores(sentence)
        list_of_dicts.append(adict)
    df_sentiment = pd.DataFrame(list_of_dicts)
    df_sentiment['Usuario'] = Users
    df_sentiment['Texto'] = Texto
    df_sentiment = df_sentiment[['Usuario', 'Texto', 'compound']]
    df_sentiment['compound'] = df_sentiment.compound.multiply(100)
    CSV = df_sentiment.to_csv('vaderSentiment.csv', sep=';', decimal='.', encoding='utf-8')
    return CSV

SDGS_marca = ['CS SDG c', 'CS SDG','CS SDG co', 'CS SDG conference', 'CS SDG conference 2020',
              'CS SDG conference 20', 'CS SDG confer', 'CS SDG conference ', 'CS SDG conferenc']



utils.sentiment_analyser('/home/fernan/Documents/Lynguo_def2.csv', interest=SDGS_marca)
