import stanza
import numpy as np
import pandas as pd

# Requerimientos para usar STANZA:

stanza.download('en', processors='tokenize,ner,pos,lemma,mwt') # se descarga el modelo en inglés y los procesadores necesarios
nlp = stanza.Pipeline('en', processors='tokenize,ner,pos,lemma,mwt')  # se inicializa el Pipeline con el idioma y procesadores

# Función extractora de palabras


def extractwords(doc):
    stop_words = ['#citizenscience', 'citizenscience', 'rt', 'citizen', 'science', 'citsci','cienciaciudadana']
    words = []
    for sent in doc.sentences:
        for word in sent.words:
            if word.upos in ('NOUN', 'ADJ', 'VERB') and word.text.startswith("\\") == False:
                words.append(word.text)

    wordslist = [x.lower() for x in words]

    #Se descartan las palabras relacionadas con ciencia ciudadana
    wordslist = [word for word in wordslist if word not in stop_words]

    wordslist = np.unique(wordslist, return_counts=True)

    return wordslist

df = pd.read_csv('/home/fernan/Documents/Lynguo_def2.csv', sep=';', encoding='utf-8', error_bad_lines=False)
subset = df['Texto']
subset = subset.astype(str)
texto = '\n\n'.join(subset.to_list())
doc = nlp(texto)
wordlist = extractwords(doc)
lista = [x for x in wordlist[0]]
lista2 = [x for x in wordlist[1]]
data_tuples = list(zip(lista,lista2))
words_df = pd.DataFrame(data_tuples,columns=['Word', 'Ntimes'])
words_df.to_csv('wordlist.csv', sep=';', index=False, encoding='utf-8')


