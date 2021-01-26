# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 10:02:33 2020

@author: ael-k
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import pickle
import numpy as np
import sklearn as sk

##### load dictionary
d = pickle.load(open("dictionary.pkl","rb"))

#####load preprocessed data
data = pd.read_csv('data.csv')
data = data.drop('new',1)

##### split the dataset / x data / y labels
x_train,x_test,y_train,y_test = train_test_split(data['text'], data['label'], test_size=0.25, random_state=7)
useless,x_validation,useless,y_validation = train_test_split(x_train, y_train, test_size=0.3, random_state=7)

##### initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english',max_features=1000,max_df=0.7)

##### fit and transform train and test set into a matrix of TF-IDF features
tfidf_train = tfidf_vectorizer.fit_transform(x_train.values.astype('str'))
tfidf_validation = tfidf_vectorizer.transform(x_validation.values.astype('str'))
tfidf_test = tfidf_vectorizer.transform(x_test.values.astype('str'))
#tfidf_train = np.asmatrix(tfidf_train)
#tfidf_validation = np.asmatrix(tfidf_validation)
#tfidf_test = np.asmatrix(tfidf_test)
tfidf_train = sk.preprocessing.normalize(tfidf_train.T)
tfidf_validation = sk.preprocessing.normalize(tfidf_validation.T)
tfidf_test = sk.preprocessing.normalize(tfidf_test.T)

##### define batch generator
seed = 7
np.random.seed(seed)
def batch_generator(X_data, y_data, batch_size):
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    index = np.arange(np.shape(y_data)[0])
    while 1:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X_data[index_batch,:].toarray()
        y_batch = y_data[y_data.index[index_batch]]
        counter += 1
        yield X_batch,y_batch
        if (counter > number_of_batches):
            counter=0

##### build neural network model
#input_dim = tfidf_train.shape[1]  # Number of features
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=1000))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit_generator(generator=batch_generator(tfidf_train, y_train, 32),
                    epochs=5, validation_data=(tfidf_validation, y_validation),
                    steps_per_epoch=tfidf_train.shape[0]/32)

loss, accuracy = model.evaluate(tfidf_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(tfidf_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

