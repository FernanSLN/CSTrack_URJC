import sys
sys.path.insert(1, '/home/fernan/Documents/Proyectos/CSTrack-URJC')
from utils import utils

subsetRT = utils.sentiment_resultsRT('/home/fernan/Documents/Proyectos/CSTrack-URJC/SDGS/vaderSentiment.csv', n=10)

subset = utils.sentiment_results('/home/fernan/Documents/Proyectos/CSTrack-URJC/SDGS/vaderSentiment.csv', n=10)

utils.combined_vader(subsetRT, subset, n=10)
