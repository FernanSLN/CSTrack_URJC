from utils.utils import plotbarchart
import pandas as pd

df = pd.read_csv('wordlist.csv', sep=';', encoding='latin-1')

df = df.sort_values('Ntimes').copy()

sortedNumberWords, sortedWords = list(df.Ntimes), list(df.Word)
sortedNumberWords = sorted(sortedNumberWords, reverse=True)
sortedWords = sorted(sortedWords, reverse=True)


plotbarchart(10, sortedWords, sortedNumberWords, 'Top 10 palabras más utilizadas',
                   'Palabras', 'Nº de veces')

for row in df:
        idx= df[df['Word'].str.match('#')]
        df2 = df.drop(idx.index)

df2 = df2.sort_values('Ntimes')

sortedNW, sortedW = list(df2.Ntimes), list(df2.Word)
sortedNW = sorted(sortedNW, reverse=True)
sortedW = sorted(sortedW, reverse =True)


plotbarchart(10, sortedW, sortedNW, 'Top 10 palabras más utilizadas',
                   'Palabras', 'Nº de veces')

