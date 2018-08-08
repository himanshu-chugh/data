import pandas as pd
import torch
from v1_fun import V1
"""
df1=pd.read_csv("train.csv")
for i in range(len(df1["Sex"])):
    if(df1["Sex"][i]=='male'):
        df1["Sex"][i]=1
    else:
        df1["Sex"][i]=0

df1.to_csv('out.csv')
"""
V1()
