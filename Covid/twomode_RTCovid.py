import utils
from webweb import Web
from covid_keywords import covid

G = utils.get_twomodeRT('/home/fernan/Documents/Lynguo_def2.csv', keywords=covid)

web = Web(title="main graph", nx_G=G)
web.display.gravity = 1

web.show()