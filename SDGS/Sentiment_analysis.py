import sys
sys.path.insert(1, '/home/fernan/Documents/Proyectos/CSTrack-URJC')
import utils
import pandas as pd
import re

subsetRT = utils.sentiment_resultsRT('/home/fernan/Documents/Proyectos/CSTrack-URJC/SDGS/vaderSentiment.csv', n=10)

subset = utils.sentiment_results('/home/fernan/Documents/Proyectos/CSTrack-URJC/SDGS/vaderSentiment.csv', n=10)

utils.combined_vader(subsetRT, subset, n=10)
