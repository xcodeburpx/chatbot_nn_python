#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 16:10:23 2017

@author: kinshiryuu-burp
"""

import pickle
import numpy as np
from nltk import word_tokenize
import string

from keras.models import load_model

from NGRAM_usage_test import read_data, clear_sentences, no_text_punkt
from NGRAM_usage_test import padding_sents, sent_to_idx


def prediction_discrete(example_data):
    for i, example in enumerate(example_data):
        for j, ex in enumerate(example):
            if ex > 0.5:
                example_data[i][j] = 1.
            else:
                example_data[i][j] = 0.
                

def extract_information(sentence, predicts):
    extract = []
    for i, predict in enumerate(predicts):
        if predict == 1:
            extract.append(sentence[i+1])
    return " ".join(extract)
        

input_path = "./NGRAM_example[length_32_vector_100]_|Wed May 10 20:12:28 2017|_"
model_path = "./NGRAM_example[length_32_vector_100]_|Wed May 10 20:12:28 2017|_/model_dense.h5"

sentences, word2index, index2word, data_path = read_data(input_path)

del sentences, data_path

model = load_model(model_path)

def extract_data(sentences):
    sent = no_text_punkt([sentences])
    sent = clear_sentences(sent)
    padding_sents(sent, 32)
    sent = sent[0]
    sent_idx = sent_to_idx(word2index, sent)
    
    sent_idx = np.reshape(sent_idx, (1, sent_idx.shape[0], sent_idx.shape[1], sent_idx.shape[2]))
    predicts = model.predict(sent_idx)
    prediction_discrete(predicts)
    
    predicts = np.reshape(predicts, (predicts.shape[1],))
    
    extraction = extract_information(sent, predicts)
    
    return extraction
"""
sentences = input("Example: ")

sent = no_text_punkt([sentences])
sent = clear_sentences(sent)
padding_sents(sent, 32)
sent = sent[0]
sent_idx = sent_to_idx(word2index, sent)
    
sent_idx = np.reshape(sent_idx, (1, sent_idx.shape[0], sent_idx.shape[1], sent_idx.shape[2]))
predicts = model.predict(sent_idx)
prediction_discrete(predicts)
    
predicts = np.reshape(predicts, (predicts.shape[1],))
    
extraction = extract_information(sent, predicts)
print(extraction)
"""
