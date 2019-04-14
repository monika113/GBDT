#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:39:44 2019

@author: monika

train and test GBDT model on dataset '20newsgroups'
"""

from data_processing import clean_text
from data_processing import load_data


#obtain train/test set
train_num, test_num, x_train, y_train, x_test, y_test, class_names = load_data('20_newsgroups', 
                                                                             split = 0.3)

for i in range(train_num):
    x_train[i] = clean_text(x_train[i])

for i in range(test_num):
    x_test[i] = clean_text(x_test[i])


#TFIDF
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
#Covert train/test set into a matrix of token counts
count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(x_train)
x_test_counts = count_vect.transform(x_test)
# calculate TFIDF
tf_trans = TfidfTransformer()
x_train_tfidf = tf_trans.fit_transform(x_train_counts)
train_weight = x_train_tfidf.toarray()
x_test_tfidf = tf_trans.transform(x_test_counts)
test_weight=x_test_tfidf.toarray()

#feature selection
from sklearn.feature_selection import SelectKBest, chi2
ch2 = SelectKBest(chi2, k=100)
x_train = ch2.fit_transform(train_weight, y_train)
x_test = ch2.transform(test_weight)

#train and test GDDT model
from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier(n_estimators = 25,
                                learning_rate = 0.1, max_features = "auto", max_depth = 3, subsample = 0.8)
gbdt.fit(x_train, y_train)
y_test_pre = gbdt.predict(x_test)

#evaluate performance(presicion,recall,f1-score)
from sklearn.metrics import classification_report
with open("result.txt", 'w+') as f:
    print(classification_report(y_test, y_test_pre, target_names=class_names),file=f)
f.close()