#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:01:38 2017

@author: kinshiryuu-burp
"""

import os
import time
import numpy as np

import sentence_extraction_text_2 as se_2

from DL_extraction_class import SE_Network

def measure_time(func):
    def decorator(*args):
        print("$"*40)
        print("\nSTARTED MEASURING TIME of {}\n".format(func.__name__))
        print("$"*40)
        start = time.time()
        products = func(*args)
        print("$"*40)
        print("\nElapsed time of {name:s} : {time:0.4f} seconds\n".format(name=func.__name__, time=time.time() - start))
        print("$"*40)
        return products
    return decorator


@measure_time
def dl_main(whole_text, fd, probabilities, max_len, len_dict, sentences, sentences_1, pallete, train_data, train_target, model):
    
    network = SE_Network(train_data.shape, hidden_size=50, hidden_count=2,
                         layer_type="lstm", activation="relu", loss="mse", optimizer="rmsprop")
    network.create_model()
    network.teach(train_data, train_target)
    
    return network
    
if __name__ == "__main__":
    
    print("DATA CREATING LALALALALA\n\n")
    whole_text, fd, probabilities, max_len, len_dict, sentences, sentences_1, pallete, train_data, train_target, model = se_2.execute_main()
    print("\nTESTING NETWORK\n\n")
    network = dl_main(whole_text, fd, probabilities, max_len, len_dict, sentences, sentences_1, pallete, train_data, train_target, model)
    print("OWARIMASU!!\n\n")
    
    
""" CHECKED COMBINATIONS"""    
""" SE_Network: data_shape - (58958, 250), hidden_size - 50, hidden_count - 2, layer_type - dense, activation - relu, loss - mse, optimizer - rmsprop """
""" SE_Network: data_shape - (58958, 250), hidden_size - 50, hidden_count - 2, layer_type - dense, activation - lstm, loss - mse, optimizer - rmsprop """