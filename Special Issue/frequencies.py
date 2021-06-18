import sys
sys.path.insert(2, '/home/fernan/Documents/Proyectos/CSTrack-URJC')
from utils import tfidf_wordcloud
from DataFrame import df



with open("/home/fernan/Documents/Lista de SDGS.txt", "r") as file:
    lines = file.readlines()
    sdgs_keywords = []
    for l in lines:
        sdgs_keywords.append(l.replace("\n", ""))

tfidf_wordcloud(df, keywords=sdgs_keywords)