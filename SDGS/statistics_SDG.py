import sys
sys.path.insert(1, '/home/fernan/Documents/Proyectos/CSTrack-URJC')
import utils
import pandas as pd


# Listado de Marcas de SDGS:

SDGS_marca = ['CS SDG c', 'CS SDG','CS SDG co', 'CS SDG conference', 'CS SDG conference 2020',
              'CS SDG conference 20', 'CS SDG confer', 'CS SDG conference ', 'CS SDG conferenc']

# Filtrado del df para dejar solo tweets sobre SDGS:

df = pd.read_csv('/home/fernan/Documents/Lynguo_def2.csv', sep=';', encoding='utf-8', error_bad_lines=False)
df = df[['Fecha', 'Texto', 'Usuario', 'Opinion', 'Marca', 'Impacto']]

df_SDGS = utils.filter_by_interest(df, SDGS_marca)

print('La cantidad de tweets obtenidos es:', len(df_SDGS))

