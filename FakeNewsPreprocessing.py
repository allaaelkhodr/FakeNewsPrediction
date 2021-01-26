# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 16:18:57 2020

@author: ael-k
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

##### help functions
def PosNormal(mean, sigma):
    x = np.random.normal(mean,sigma,1)
    return(x if x>=0 else PosNormal(mean,sigma))

##### read data
fake_data = pd.read_csv('fake.csv')
true_data = pd.read_csv('true.csv')
fake_data['label'] = 1
true_data['label'] = 0

#_train,fake_data,y_train,y_test=train_test_split(fake_data, fake_data, test_size=0.01, random_state=7)
#fake_data.to_csv('Fake_split.csv')

# =============================================================================
# ##### calculate article length
# true_article_length = true_data['text'].str.split().str.len()
# fake_article_length = fake_data['text'].str.split().str.len()
# print(tabulate([['true', true_article_length.mean(), true_article_length.std()],
#                 ['fake', fake_article_length.mean(), fake_article_length.std()]],
#                 headers=['article' ,'Number of Words - mean', 'Standard Deviation']),
#                 '\n')
# 
# plt.figure('1')
# true_article_dist = np.random.normal(true_article_length.mean(), true_article_length.std(), size=1000)
# plt.xlabel('text')
# plt.title('Number of Words in Real News Article')
# count, bins, ignored = plt.hist(true_article_dist, 30, density=True)
# plt.plot(bins, 1/(true_article_length.std() * np.sqrt(2 * np.pi)) *
#                np.exp( - (bins - true_article_length.mean())**2 / (2 * true_article_length.std()**2) ),
#          linewidth=2, color='r')
# plt.show()
# 
# plt.figure('2')
# fake_article_dist = np.random.normal(fake_article_length.mean(), fake_article_length.std(), size=1000)
# plt.xlabel('text')
# plt.title('Number of Words in Fake News Article')
# count, bins, ignored = plt.hist(fake_article_dist, 30, density=True)
# plt.plot(bins, 1/(fake_article_length.std() * np.sqrt(2 * np.pi)) *
#                np.exp( - (bins - fake_article_length.mean())**2 / (2 * fake_article_length.std()**2) ),
#          linewidth=2, color='r')
# plt.show()
# 
# ##### calculate title length
# true_title_length = true_data['title'].str.split().str.len()
# fake_title_length = fake_data['title'].str.split().str.len()
# print(tabulate([['true', true_title_length.mean(), true_title_length.std()],
#                 ['fake', fake_title_length.mean(), fake_title_length.std()]],
#                 headers=['title' ,'Number of Words - mean', 'Standard Deviation']),
#                 '\n')
# 
# plt.figure('3')
# true_title_dist = np.random.normal(true_title_length.mean(), true_title_length.std(), size=1000)
# plt.xlabel('text')
# plt.title('Number of Words in Real News Titles')
# count, bins, ignored = plt.hist(true_title_dist, 30, density=True)
# plt.plot(bins, 1/(true_title_length.std() * np.sqrt(2 * np.pi)) *
#                np.exp( - (bins - true_title_length.mean())**2 / (2 * true_title_length.std()**2) ),
#          linewidth=2, color='r')
# plt.show()
# 
# plt.figure('4')
# fake_title_dist = np.random.normal(fake_title_length.mean(), fake_title_length.std(), size=1000)
# plt.xlabel('text')
# plt.title('Number of Words in Fake News Titles')
# count, bins, ignored = plt.hist(fake_title_dist, 30, density=True)
# plt.plot(bins, 1/(fake_title_length.std() * np.sqrt(2 * np.pi)) *
#                np.exp( - (bins - fake_title_length.mean())**2 / (2 * fake_title_length.std()**2) ),
#          linewidth=2, color='r')
# plt.show()
# =============================================================================

##### import stopwords
stop_words = stopwords.words('english')

##### remove stop words from articles
number_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
tokenizer = RegexpTokenizer(r'\w+')
for x in range(0, fake_data['text'].size):
    if x%100 == 0:
        print(x)
    for stop_word in stop_words:
        fake_data['text'][x] = fake_data['text'][x].replace(" " + stop_word + " ", " ")   
    for number in number_list:
        fake_data['text'][x] = fake_data['text'][x].replace(number, " ")

for x in range(0, true_data['text'].size):
    for stop_word in stop_words:
        true_data['text'][x] = true_data['text'][x].replace(" " + stop_word + " ", " ")   
    for number in number_list:
        true_data['text'][x] = true_data['text'][x].replace(number, " ")

##### tokenizing
for x in range(0, fake_data['text'].size):  
        if x%100 == 0:
            print(x)      
        fake_data['text'][x] = tokenizer.tokenize(fake_data['text'][x])    

for x in range(0, true_data['text'].size):        
        true_data['text'][x] = tokenizer.tokenize(true_data['text'][x])
   
##### transform words into root form
lemmatizer = WordNetLemmatizer()
for x in range(0, fake_data['text'].size):
    for y in range(0, len(fake_data['text'][x])):
        if x%100 == 0:
            print(x)
        fake_data['text'][x][y] = lemmatizer.lemmatize(fake_data['text'][x][y], 'v')

for x in range(0, true_data['text'].size):
    for y in range(0, len(true_data['text'][x])):
        true_data['text'][x][y] = lemmatizer.lemmatize(true_data['text'][x][y], 'v')

##### transform data in df to a list
for x in range(0, fake_data['text'].size):
    if x%100 == 0:
        print(x)
    fake_data['text'][x] = ', '.join(fake_data['text'][x])

for x in range(0, true_data['text'].size):
    if x%100 == 0:
        print(x)
    true_data['text'][x] = ', '.join(true_data['text'][x])

##### concatenate fake and true data
data = pd.concat([fake_data, true_data])
data['new']=range(0,data['label'].size)
data = data.set_index('new')

##### save preprocessed data as csv file
data.to_csv('data.csv')