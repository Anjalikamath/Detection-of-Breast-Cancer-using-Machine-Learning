import sys
import math
import numpy as np
import pandas as pd
filename=sys.stdin
import csv

#reading cleaned.csv and sorting the columns and storing it in sorted.txt
df=pd.read_csv(filename,index_col=False,header=None)
#normal_df=(df-df.min())/(df.max()-df.min())
#f_df=normal_df.apply(lambda x : x.sort_values().values)
f_df=df.apply(lambda x : x.sort_values().values)
sorted_csv=f_df.to_csv(r'/home/anjali/Desktop/sem5/ml_project/sorted.txt',index=None,header=None)
