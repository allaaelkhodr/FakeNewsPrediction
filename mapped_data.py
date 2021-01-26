# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 11:07:00 2020

@author: ael-k
"""

import pandas as pd
import pickle
import math

##### load dictionary
d = pickle.load(open("dictionary.pkl","rb"))

##### load preprocessed data
data = pd.read_csv('data.csv')
data = data.drop('new',1)
for article in range(0,data['text'].size):
    if isinstance(data['text'][article], str)==False:
        if math.isnan(data['text'][article])==True:
            continue
    data['text'][article] = data['text'][article].split(',')
    print(article)
    
##### map data
for x in range(data['text'].size):
    list_df = pd.DataFrame(data['text'][x],columns=['Column_Name']).replace({"Column_Name": d})
    data['text'][x] = list_df.values.tolist()
    print(x)
    
##### save data as csv file
data.to_csv('mapped_data.csv')