import utils
from webweb import Web


G = utils.get_twomodeRT('/home/fernan/Documents/Lynguo_April21.csv', keywords=['Horizon 2020', 'H 2020', 'H2020'])

web = Web(title="main graph", nx_G=G)
web.display.gravity = 1
web.show()
