import modin.pandas as pd
import ray
# In windows import dask and initiate dask.

ray.init()

df = pd.read_csv("/home/fernan/Documents/Lynguo_May21.csv", sep=';', encoding='utf-8', decimal=',', error_bad_lines=False)

print(df)
