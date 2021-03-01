import utils

most_common = utils.most_common('/home/fernan/Documents/Lynguo_def2.csv', 10)

names = [item[0] for item in most_common]

numbers = [item[1] for item in most_common]

utils.plotbarchart(10, names, numbers, 'top 10 palabras mas usadas', 'palabras', 'numero de veces')


utils.most_commonwc('/home/fernan/Documents/Lynguo_def2.csv')