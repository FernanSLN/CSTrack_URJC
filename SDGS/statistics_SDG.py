import utils
import pandas as pd

# List of marks of SDGS:

SDGS_marca = ['CS SDG c', 'CS SDG','CS SDG co', 'CS SDG conference', 'CS SDG conference 2020',
              'CS SDG conference 20', 'CS SDG confer', 'CS SDG conference ', 'CS SDG conferenc']

# Filter by sdgs:

df = pd.read_csv('/home/fernan/Documents/Lynguo_def2.csv', sep=';', encoding='utf-8', error_bad_lines=False,
                 decimal=',', low_memory=False)
df = df[['Texto', 'Usuario', 'Opinion', 'Marca', 'Impacto']]

df_SDGS = utils.filter_by_interest(df, SDGS_marca)

df_SDGS = df_SDGS.dropna()

# Calculate mean opinion and impact:

print('La cantidad de tweets obtenidos es:', len(df_SDGS))



df_SDGS['Opinion'] = df_SDGS['Opinion'].replace(',','.', regex=True).astype(float)

print(df['Opinion'])

print('La media de opinión es de:', round((df_SDGS['Opinion'].mean()),2))
print('La media de impacto es de:', round((df_SDGS['Impacto'].mean()),2))

print('La mediana(estadística) de opinión es de:', round((df_SDGS['Opinion'].median()),2))
print('La mediana(estadística) de impacto es de:', round((df_SDGS['Impacto'].median()),2))

print('La desviación estándar de opinión es de:', round((df_SDGS['Opinion'].std()),2))
print('La desviación estándar de impacto es de:', round((df_SDGS['Impacto'].std()),2))

#utils.impact_opinionRT('/home/fernan/Documents/Lynguo_def2.csv', interest=SDGS_marca, Opinion=True, n=10)

df_statistics = utils.dataframe_statistics('/home/fernan/Documents/Lynguo_def2.csv')

print(df_statistics)
