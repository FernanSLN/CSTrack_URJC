import modin.pandas as mdpd
import ray
import sys
sys.path.insert(1, '/home/fernan/Documents/Proyectos/CSTrack-URJC')

ray.init()

# In windows import dask and initiate dask.
#import os
#os.environ["MODIN_ENGINE"] = "dask"
#import modin.pandas as pd

df = mdpd.read_csv("/home/fernan/Documents/Lynguo_22June.csv", sep=';', encoding='utf-8', decimal=',',
                 error_bad_lines=False)

