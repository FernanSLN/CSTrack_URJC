import sys
sys.path.insert(1, '/home/fernan/Documents/Proyectos/CSTrack-URJC')
import utils
import pandas as pd
import seaborn as sns
import statistics as stats
import re
import numpy as np

# Listado de Marcas de SDGS:

SDGS_marca = ['CS SDG c', 'CS SDG','CS SDG co', 'CS SDG conference', 'CS SDG conference 2020',
              'CS SDG conference 20', 'CS SDG confer', 'CS SDG conference ', 'CS SDG conferenc']

# Filtrado del df para dejar solo tweets sobre SDGS:

df = pd.read_csv('/home/fernan/Documents/Lynguo_def2.csv', sep=';', encoding='utf-8', error_bad_lines=False,
                 decimal=',', dtype={'Impacto':'float64'})
df = df[['Fecha', 'Texto', 'Usuario', 'Opinion', 'Marca', 'Impacto']]

df_SDGS = utils.filter_by_interest(df, SDGS_marca)

df_SDGS = df_SDGS.dropna()

# Calculamos la cantidad de tweets y la media de opinión e impacto:

print('La cantidad de tweets obtenidos es:', len(df_SDGS))



df_SDGS['Opinion'] = (df_SDGS['Opinion'].replace(',','.', regex=True).astype(float))


print('La media de opinión es de:', round((df_SDGS['Opinion'].mean()),2))
print('La media de impacto es de:', round((df_SDGS['Impacto'].mean()),2))

print('La mediana(estadística) de opinión es de:', round((df_SDGS['Opinion'].median()),2))
print('La mediana(estadística) de impacto es de:', round((df_SDGS['Impacto'].median()),2))

print('La desviación estándar de opinión es de:', round((df_SDGS['Opinion'].std()),2))
print('La desviación estándar de impacto es de:', round((df_SDGS['Impacto'].std()),2))

# Separamos los RTs del resto de tweets y vemos que usuarios tienen mayor impacto y opinión
# df_SDGS.sort_values('Opinion', ascending=False))

df_SDGSRT = df_SDGS[['Usuario','Texto', 'Impacto']]
idx = df_SDGSRT['Texto'].str.contains('RT @', na=False)
df_SDGSRT = df_SDGSRT[idx]  # Se seleccionan sólo las filas con RT
df_SDGSRT = df_SDGSRT.sort_values('Impacto', ascending=False)
df_SDGSRT = df_SDGSRT.drop_duplicates(subset='Texto', keep='first')
subset = df_SDGSRT[['Texto', 'Impacto']]
retweets = [list(x) for x in subset.to_numpy()]

names = [item[0] for item in retweets]
numbers = [item[1] for item in retweets]


#utils.plotbarchart(10, names, numbers, 'Top 10 tweets con mayor impacto', 'Tweets', 'Opinion' )

#subset[:50].to_csv('top50 tweets retuiteados por impacto.csv', sep=';', index=False, decimal='.', encoding='utf-8', )

arrobas = []
for row in retweets:
    reg = re.search('@(\w+)', row[0])
    if reg:
        matchRT = reg.group(1)
        row[0] = matchRT
        arrobas.append(row)

df_arrobas = pd.DataFrame(arrobas, columns=['User', 'Impact'])
df_arrobas = df_arrobas.groupby('User', as_index=False).mean()
df_arrobas = df_arrobas.sort_values('Impact', ascending=False)
users = df_arrobas['User']
impacto = df_arrobas['Impact']
utils.plotbarchart(10, users, impacto, 'Top 10 Usuarios con mayor impacto', 'Usuario', 'Impacto')


