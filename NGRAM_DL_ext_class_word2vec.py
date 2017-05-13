#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 14:31:32 2017

@author: kinshiryuu-burp
"""


import numpy as np
import time
import tensorflow as tf

import pickle

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, GaussianNoise, Reshape
from keras.layers import LSTM, GRU
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import CSVLogger, ModelCheckpoint

from NGRAM_usage_test import read_data, create_data

input_path = "./PROJECT_DUMP/NGRAM_example[length_32_vector_100]_|Wed May 10 22:02:47 2017|_"

sentences, word2index, index2word, save_path= read_data(input_path)

del index2word

test_data_path = input_path + "/" + "test_data_dense.pkl"
test_target_path = input_path + "/" + "test_target_dense.pkl"

class SE_Network:
    
    def __init__(self, data_shape, target_shape, hidden_size, hidden_count,  layer_type, activation, loss, optimizer):
        
        assert(data_shape != (0,0,0,0)), "Shape of data must be different than (0,0,0)"
        assert(target_shape != (0,0)), "Shape of target must be different than (0,0)"
        assert(hidden_size >= 1), "Size of hidden layer must be >= 0"
        assert(hidden_count >= 1), "Number of layers must be >= 0"
        assert(layer_type.lower() in ["dense", "lstm"]), "Layer type must be string of 'dense' or 'lstm'"
        assert(type(activation) == type('')), "Activation function must be a name in string"
        assert(type(loss) == type('')), "Loss function must be a name in string"
        assert(type(optimizer) == type('')), "Optimizer function must be a name in string"
        
        
        self.data_shape = data_shape
        self.target_shape = target_shape
        self.hidden_size = hidden_size
        self.hidden_count = hidden_count
        self.layer_type = layer_type
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        print("Train_target shape:",target_shape)
        
    def __repr__(self):
        repres = "SE_Network: data_shape - {}, target_shape= {} hidden_size - {}, hidden_count - {}, layer_type - {}, activation - {}, loss - {}, optimizer - {}".format(
                self.data_shape, self.target_shape, self.hidden_size, self.hidden_count, self.layer_type, self.activation, self.loss, self.optimizer)
        return repres
    
    def __str__(self):
        return repr(self)
    
    def create_model(self):
        self.model = Sequential()
        
        if self.layer_type.lower() == "dense":
            for i in range(self.hidden_count):
                if i == 0:
                    self.model.add(Dense(self.hidden_size, kernel_initializer="he_uniform", input_shape=(self.data_shape[1], self.data_shape[2], self.data_shape[3],)))
                    self.model.add(Activation(self.activation))
                    self.model.add(Dropout(0.3))
                else:
                    self.model.add(Dense(self.hidden_size, kernel_initializer="he_uniform"))
                    self.model.add(Activation(self.activation))
                    self.model.add(Dropout(0.3))
            self.model.add(Reshape((-1,)))
            self.model.add(Dense(self.target_shape[-1], activation='sigmoid'))
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
            
                    
        if self.layer_type.lower() == "lstm":  
            if self.hidden_count == 0:
                self.model.add(Reshape((self.data_shape[1], self.data_shape[2]*self.data_shape[3]), input_shape=(self.data_shape[1], self.data_shape[2],self.data_shape[3])))
                self.model.add(LSTM(self.hidden_size, init='normal'))
                self.model.add(Activation(self.activation))
                self.model.add(Dropout(0.3))
            else:
                for i in range(self.hidden_count):
                    if i == 0:
                        self.model.add(Reshape((self.data_shape[1], self.data_shape[2] * self.data_shape[3]), input_shape=(self.data_shape[1], self.data_shape[2],self.data_shape[3])))
                        self.model.add(LSTM(self.hidden_size, init='normal', return_sequences=True))
                        self.model.add(Activation(self.activation))
                        self.model.add(Dropout(0.3))
                    
                    elif i == self.hidden_count - 1:
                        
                        self.model.add(LSTM(self.hidden_size, init='normal', return_sequences=False))
                        self.model.add(Activation(self.activation))
                        self.model.add(Dropout(0.3))
                        
                    
                    else:
                        self.model.add(LSTM(self.hidden_size, init='normal', return_sequences=True))
                        self.model.add(Activation(self.activation))
                        self.model.add(Dropout(0.3))
            
            self.model.add(Dense(self.target_shape[-1], activation='sigmoid'))
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        
        
    def teach(self, batch_size = 50000):     
        print("DATA GENERATION\n")
        iters = int(np.ceil(len(sentences)/batch_size))
        
        generate_data = create_data(word2index, sentences, batch_size)
        
        test_data, test_target = next(generate_data)
        
        print("\nIterations", iters)
        
        time.sleep(1)
        for i in range(iters-1):
            print("\n",(i+1),"Iteration\n")
            try:
                train_data, train_target = next(generate_data)
                self.model.fit(train_data, train_target, batch_size=128, epochs=5,
                               verbose=1, validation_data=(test_data, test_target))
            except StopIteration:
                generate_data = create_data(word2index, sentences, batch_size)
        
        print("\nSAVING DATA AND TARGET TEST\n")
        with open(test_data_path, "wb") as data:
            pickle.dump(test_data, data)
            
        with open(test_target_path, "wb") as target:
            pickle.dump(test_target, target)
        
    def predict(self, predict_data):
        predicts = self.model.predict(predict_data)
        
        return predicts
    
    def save(self):
        to_save_path = save_path+"/model_"+self.layer_type+".h5"
        self.model.save(to_save_path)
            
        
            
        