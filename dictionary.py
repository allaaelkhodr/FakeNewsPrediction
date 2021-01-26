# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 16:32:01 2020

@author: ael-k
"""
import pandas as pd
import math
x = float('nan')
math.isnan(x)

#####load preprocessed data
data = pd.read_csv('data.csv')
data.drop('new',1)
counter = 0
for article in range(0,data['text'].size):
    if isinstance(data['text'][article], str)==False:
        if math.isnan(data['text'][article])==True:
            continue
    data['text'][article] = data['text'][article].split(',')
    counter+=1
    print(counter)

d = dict()
i = 0
counter = 0
articles = data['text']
for article in articles:
    if isinstance(article, list)==False:
        if math.isnan(article)==True:
            continue
    for word in article:
        if word not in d:
            d[word] = i
            i += 1
            
import pickle
f = open("dictionary.pkl","wb")
pickle.dump(d,f)
f.close()