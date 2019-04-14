#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:26:33 2019

@author: monika

load_data:
    load dataset '20newspapers'
    generate raw train/test datasets

clean_text: 
    pre processing for dataset '20newspapers':
        lowercase
        delete punctuation
        delete number
        delete continuous space
        delete stopwords
        extract stem
"""

import re
import nltk
from nltk import PorterStemmer, word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
import string
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


# generate raw train/test datasets
def load_data(filename,split = 0.3):
    #load data
    dataset = load_files(filename,
            load_content = True,
            encoding = 'latin1',
            decode_error = 'strict',
            shuffle = True)
    #split train set and test set
    test_num = int(split*len(dataset.data))
    train_num = len(dataset.data)-test_num
    x_train = dataset.data[:train_num]
    y_train = dataset.target[:train_num]
    x_test = dataset.data[train_num:]
    y_test = dataset.target[train_num:]
    class_name = dataset.target_names
    return train_num, test_num, x_train, y_train, x_test, y_test, class_name

#return a string of clean text
def clean_text(text):
    #lowercase
    text = text.lower()
    #delete punctuation
    tran = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    text = text.translate(tran)
    #delete number
    text = re.sub("\d+", " ", text)
    #delete continuous space
    text = re.sub(r'\s+', " ", text)
    #participle
    tokens = word_tokenize(text)
    #delete stopwords
    tokens = [w for w in tokens if not w in stopwords.words('english')] 
    #extract stem
    stemmer = PorterStemmer()
    tokens_stemmed = []
    text="";
    for w in tokens:
        tokens_stemmed.append(stemmer.stem(w))
    #combine
    for w in tokens_stemmed:
        text += " "+w
    return text


