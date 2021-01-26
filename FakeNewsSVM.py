# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 08:41:19 2020

@author: ael-k
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import sklearn as sk
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

#####load preprocessed data
data = pd.read_csv('data.csv')
data = data.drop('new',1)

##### split the dataset
x_train,x_test,y_train,y_test = train_test_split(data['text'], data['label'], test_size=0.3, random_state=7)

##### initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english',max_df=0.7)

##### fit and transform train and test set into a matrix of TF-IDF features
tfidf_train = tfidf_vectorizer.fit_transform(x_train.values.astype('str'))
tfidf_test = tfidf_vectorizer.transform(x_test.values.astype('str'))
tfidf_train = sk.preprocessing.normalize(tfidf_train)
tfidf_test = sk.preprocessing.normalize(tfidf_test)
# =============================================================================
# ##### gridSearch-crossValidation for optimal parameters
# nfolds = 3
# kernels = ['rbf', 'linear', 'sigmoid', 'poly']
# Cs = [0.001, 0.01, 0.1, 1, 10]
# gammas = [0.001, 0.01, 0.1, 1]
# param_grid = {'kernel': kernels,'C': Cs, 'gamma' : gammas}
# grid_search = GridSearchCV(svm.SVC(decision_function_shape='ovo'), param_grid, cv=nfolds)
# grid_search.fit(tfidf_train, y_train)
# optimal_parameter = grid_search.best_params_
# best_score = grid_search.best_score_
# print("\nbest accuracy in the validationset:", best_score)
# =============================================================================

##### train model
optimal_parameter = {'kernel': 'linear','C': 10, 'gamma' : 0.001}
clf = svm.SVC(kernel=optimal_parameter['kernel'], C=optimal_parameter['C'], gamma=optimal_parameter['gamma'], decision_function_shape='ovo')
clf.fit(tfidf_train, y_train)

##### test model
predicted_labels = clf.predict(tfidf_test)
accuracy = metrics.accuracy_score(y_test, predicted_labels)
confusion_matrix = metrics.confusion_matrix(y_test, predicted_labels)
classification_report = metrics.classification_report(y_test, predicted_labels)

##### print results
print("accuracy:", accuracy)
print("number of training data:", len(x_train))
print("number of test data:", len(x_test))
print('\nClassification Report:\n', classification_report)

plot_confusion_matrix(conf_mat=confusion_matrix,show_normed=True,colorbar=True)
plt.show()
