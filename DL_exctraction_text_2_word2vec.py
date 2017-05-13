#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 10:55:04 2017

@author: kinshiryuu-burp
"""

import os
import pickle
import time
import numpy as np

import POS_usage_test_2 as pos_test_2

from DL_extraction_class_word2vec import SE_Network

def measure_time(func):
    def decorator(*args):
        print("$"*40)
        print("\nSTARTED MEASURING TIME of {}\n".format(func.__name__))
        print("$"*40)
        start = time.time()
        products = func(*args)
        print("$"*40)
        meas_time = time.time() - start
        time_str = time.strftime('%H:%M:%S', time.gmtime(meas_time))
        print("\nElapsed time of {name:s} : {time:s}\n".format(name=func.__name__, time=time_str))
        print("$"*40)
        return products
    return decorator


@measure_time
def dl_main(input_shape, target_shape):
    
    network = SE_Network(input_shape, target_shape, hidden_size=200, hidden_count=3,
                         layer_type="lstm", activation="relu", loss="binary_crossentropy", optimizer="adam")
    network.create_model()
    network.teach()
    network.save()
    
    return network
    
if __name__ == "__main__":
    print("\nTESTING NETWORK\n\n")
    input_shape = (None, 31, 300)
    target_shape = (None, 31)
    network = dl_main(input_shape, target_shape)
    print("OWARIMASU!!\n\n")