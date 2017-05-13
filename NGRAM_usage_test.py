#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 21:40:09 2017

@author: kinshiryuu-burp
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
import string
import time
import pprint


import nltk
from nltk import FreqDist
from nltk import pos_tag, ngrams
from nltk.probability import ELEProbDist, SimpleGoodTuringProbDist, MLEProbDist, LaplaceProbDist, \
    LidstoneProbDist, MutableProbDist, WittenBellProbDist, KneserNeyProbDist
from nltk import word_tokenize, sent_tokenize, RegexpTokenizer, WordPunctTokenizer
from nltk.tag.stanford import StanfordNERTagger
from nltk import pos_tag, ne_chunk
from nltk.corpus import stopwords

import gensim
from gensim import corpora

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

w_tokenizer = RegexpTokenizer(r"([\w]+'[\w]+|[\w]+)")

p_tokenizer = WordPunctTokenizer()

#ner_tagger = StanfordNERTagger('/home/kinshiryuu-burp/PROJEKTS/NATURAL_LANGUAGE_PROCESSING/stanford-ner-2016-10-31/classifiers/english.all.3class.distsim.crf.ser.gz', 
#                             '/home/kinshiryuu-burp/PROJEKTS/NATURAL_LANGUAGE_PROCESSING/stanford-ner-2016-10-31/stanford-ner.jar')
vector_size = 100
length = 30

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
def extract_sentences_text(filepath):
    whole_text = []
    sentences = []
    paths = os.listdir(filepath)
    
    for i, path in enumerate(paths):
        paths[i] = filepath+"/"+path
        #print(paths[i])
        
    for i, path in enumerate(paths):
        name = "/home/kinshiryuu-burp/PROJEKTS/CHATBOT/DATABASE/SCRIPTS_TO CSV/SENTECENS_CSV/" + path.split('/')[-1]
        print(name)
        dataframe = pd.read_csv(name, header=0)
        for j in range(dataframe.shape[0]):
            sentence = dataframe.iloc[[j]]
            sentence = sentence['Dialogue/Sentence'].values
            sentence = sentence[0]
            if(type(sentence) != type('')):
                continue
            #print(sentence)
            #time.sleep(0.01)
            sentence = sent_tokenize(sentence)
            #print('done')
            words = [word_tokenize(sent) for sent in sentence]
            #if j % 100 == 0 and j != 0:
            #    print(sentence)
            sentences += sentence
            for line in words:
                new_line = ["BOS"]
                line.append('EOS')
                new_line += line
                whole_text += new_line
    
    return whole_text, sentences

def max_length(sentences):
    max_len = 0
    for sent in sentences:
        if len(word_tokenize(sent)) > max_len:
            max_len = len(word_tokenize(sent))
                
        #if len(sent) == 915:
        #    print(sent)
    max_len += 2
    return max_len

def no_punkt(lista):
    new_list = []
    for data in lista:
        if data not in string.punctuation:
            new_list.append(data)
    return new_list

def no_text_punkt(sents):
    new_sents = []
    word_sents = [word_tokenize(word) for word in sents]
    for sent in word_sents:
        clear_sent = ['bos']
        for word in sent:
            if word not in string.punctuation:
                clear_sent.append(word.lower())
        clear_sent.append('eos')
        clear_sent.append('eos')
        if len(clear_sent) != 0:
            new_sents.append(clear_sent)
    
    return new_sents

@measure_time
def clear_sentences(sentences):
    clear_sents = []
    for i, sent in enumerate(sentences):
        sentences[i] = clear_text(sentences[i])
        if len(sentences[i]) != 0:
            clear_sents.append(sentences[i])
            
    return clear_sents


def prob_dist(word_list):
    fd = FreqDist(word_list)
    funcs = [ELEProbDist, MLEProbDist, LaplaceProbDist, SimpleGoodTuringProbDist]
    probabilities = []
    for i, func in enumerate(funcs):
        print(i)
        probabilities.append(func(fd))
    
    return fd, probabilities

@measure_time
def create_palette(fd, prob, model, max_len, name_dump):
    word2index = dict()
    index2word = dict()
    items = sorted(list(fd.keys()))
    for i, key in enumerate(items):
        if key == 'eos':
            word2index[key] = (list(model.wv.word_vec(key)), np.float128(0.))
            index2word[str(list(model.wv.word_vec(key)))] = (key, np.float128(0.))
            
        elif key == 'bos':
            word2index[key] = (list(model.wv.word_vec(key)), np.float128(1.))
            index2word[str(list(model.wv.word_vec(key)))] = (key, np.float128(1.))

        elif pos_tag([key])[0][1] in ['NN', 'NNP', 'NNS', 'NNPS', 'PRP', 'PRP$', 'SYM']:
            proba = (2-np.float128(prob.prob(key)))/2
            word2index[key] = (list(model.wv.word_vec(key)), np.float128(proba))
            index2word[str(list(model.wv.word_vec(key)))] = (key,np.float128(proba))

        elif pos_tag([key])[0][1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'POS']:
            proba = (1.5-np.float128(prob.prob(key)))/2
            word2index[key] = (list(model.wv.word_vec(key)), np.float128(proba))
            index2word[str(list(model.wv.word_vec(key)))] = (key, np.float128(proba))
                
        elif pos_tag([key])[0][1] in ['CC', 'CD', 'DT', 'IN', 'LS']:
            proba = (1.-np.float128(prob.prob(key)))/2
            word2index[key] = (list(model.wv.word_vec(key)), np.float128(proba))
            index2word[str(list(model.wv.word_vec(key)))] = (key, np.float128(proba))
                
        elif pos_tag([key])[0][1] in ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']:
            proba = (1.5-np.float128(prob.prob(key)))/2
            word2index[key] = (list(model.wv.word_vec(key)), np.float128(proba))
            index2word[str(list(model.wv.word_vec(key)))] = (key, np.float128(proba))
                
        elif pos_tag([key])[0][1] in ['WDT', 'WP', 'WP$', 'WRB']:
            proba = (1.5-np.float128(prob.prob(key)))/2
            word2index[key] = (list(model.wv.word_vec(key)), np.float128(proba))
            index2word[str(list(model.wv.word_vec(key)))] = (key, np.float128(proba))
                
        else:
            proba = (1.-np.float128(prob.prob(key)))/2
            word2index[key] = (list(model.wv.word_vec(key)), np.float128(proba))
            index2word[str(list(model.wv.word_vec(key)))] = (key, np.float128(proba))
    
    eos = word2index['eos']
    bos = word2index['bos']

    word2index_path = name_dump + "/" + "word2index_" + str(max_len) + ".pkl"
    index2word_path = name_dump + "/" + "index2word_" + str(max_len) + ".pkl"
    name_eos = name_dump + "/" + "eos_" + str(max_len) + ".pkl"
    name_bos = name_dump + "/" + "bos_" + str(max_len) + ".pkl"

    
    print("\nSAVING WORD2INDEX, INDEX2WORD, EOS and BOS\n")
    time.sleep(1)
    with open(word2index_path, "wb") as fp:
        pickle.dump(word2index, fp)

    with open(index2word_path, "wb") as fp:
        pickle.dump(index2word, fp)
    
    with open(name_eos, "wb") as fe:
        pickle.dump(eos, fe)

    with open(name_bos, "wb") as fe:
        pickle.dump(bos, fe)

    print("\nSAVING FINISHED\n")
    time.sleep(1)
    return word2index, index2word, eos, bos

def sentences_reduction(sentences, max_length):
    return [sent for sent in sentences if len(word_tokenize(sent)) <= max_length and len(word_tokenize(sent)) >= 1]

def padding_sents(sentences, max_len, name_dump):
    for i in range(len(sentences)):
        while len(sentences[i]) < max_len:
            sentences[i].append('eos')

    print("\nPADDING FINISHED\n")
    print("SAVING SENTENCES\n")

    sents_path = name_dump + "/" + "sentences_" + str(max_len) + ".pkl"
    with open(sents_path, "wb") as file:
        pickle.dump(sentences, file)

    print("SENTENCES SAVED\n")

def clear_text(whole_text):
    extra_list = []
    stops = set(stopwords.words('english'))
    new_list = no_punkt(whole_text)
    new_list = [word.lower() for word in new_list]
    #new_list = [word for word in new_list if word not in stops]
    new_list = [word for word in new_list if word not in ['...', '--']]
    for word in new_list:
        if len(word) == 1:
            if word[0] not in string.punctuation:
                extra_list.append(word)
        if len(word) >= 2:
            if word[1] not in string.punctuation:
                extra_list.append(word)
    
    return extra_list

def read_data(input_path = None):
    if not input_path:
        import_path = "./PROJECT_DUMP/"

        examples = os.listdir(import_path)
        examples = sorted(examples, key=lambda x: x.split('|')[1])
    
        data_path = import_path + examples[-1]
    else:
        data_path = input_path
        
    data = sorted(os.listdir(data_path))

    search_files = ["index2word", "word2index", "sentences"]
    for ending in data:
        if search_files[0] in ending:
            i2w_path = data_path + '/' + ending
        elif search_files[1] in ending:
            w2i_path = data_path + '/' + ending
        elif search_files[2] in ending:
            sentences_path = data_path + '/' + ending
            
    with open(i2w_path, 'rb') as file:
        index2word = pickle.load(file)
        
    with open(sentences_path, 'rb') as file:
        sentences = pickle.load(file)
        
    with open(w2i_path, 'rb') as file:
        word2index = pickle.load(file)
        

    return sentences, word2index, index2word, data_path


def create_data(word2index, sentences, batch_size = 128):
    length = len(sentences)
    indices = list(range(length))
    np.random.shuffle(indices)
    sental = np.array(sentences)
    for i in range(int(np.ceil(length/batch_size))):
        train_sample = []
        train_label = []
        idx = indices[i*batch_size:(i+1)*batch_size]
        examples = sental[idx]
        for i, sent in enumerate(examples):
            sample = []
            label = []
            trigrams = ngrams(sent, 3)
            for gram in trigrams:
                eos_counter = 0
                subs = []
                probs = []
                for word in list(gram):
                    if word == "eos":
                        eos_counter += 1
                    subs.append(word2index[word][0])
                    probs.append(word2index[word][1])
                if eos_counter == 1:
                    #print("POSITIVE EOS\n")
                    probs[-1] = 0.8
                sample.append(subs)

                try:
                    if np.sum(probs) == 0.0:
                        summa = 0.0001
                        f3 = 3*np.multiply.reduce(probs)/summa
                    else:
                        f3 = 3*np.multiply.reduce(probs)/np.sum(probs)
                except RuntimeWarning:
                    if np.multiply.reduce(probs) != 0.0:
                        f3 = 1.
                    else:
                        f3 = 0.

                if f3 >= 0.75:
                    label.append(1.)
                else:
                    label.append(0.)

            if len(sample) > 0:
                #print("{}: {}".format(i,len(sample)))
                train_sample.append(sample)
                train_label.append(label)

        train_sample = np.array(train_sample)
        train_label = np.array(train_label)
        yield train_sample, train_label


@measure_time
def execute_main():
    
    filepath = "/home/kinshiryuu-burp/PROJEKTS/CHATBOT/DATABASE/SCRIPTS_TO CSV/SENTECENS_CSV"
    print("CREATE TEXT LIST AND SENTENCES\n")
    whole_text, sentences = extract_sentences_text(filepath)
    print("WHOLE TEXT", whole_text[0:10],"\n")
    print("SENTENCES", sentences[0:10], "\n")
    sentences_1 = list(sentences)
      
    print("CLEAR TEXT\n")
    whole_text = clear_text(whole_text)
    print("WHOLE TEXT", whole_text[0:10], "\n")
    
    print("PROBABILITIES AND FREQUENCIES\n")
    fd, probabilities = prob_dist(whole_text)
    print("FREQUENCIES", fd.pprint(), '\n')
    
    print("SENTENCE REDUCTION\n\n")
    sentences = [sent for sent in sentences if len(word_tokenize(sent)) <= length and len(word_tokenize(sent)) >= 1]
    print("FINISHED REDUCTION\n")
    
    print("CHECK MAX SENTENCE\n")
    max_len = max_length(sentences)
    print(max_len,'\n')

    print("CREATE DIRECTORY\n")
    localtime = time.asctime(time.localtime(time.time()))
    name_dump = "./PROJECT_DUMP/NGRAM_example[length_"+str(max_len)+"_vector_" + str(vector_size)+"]_|"+localtime+"|_"
    os.mkdir(name_dump)
    print("DIRECTORY CREATED\n")
    
    print("CLEAR SENTENCES\n")
    sentences = no_text_punkt(sentences)
    sentences = clear_sentences(sentences)
    padding_sents(sentences, max_len, name_dump)
    print("SENTENCES", *sentences[0:10], '\n')
    
    print("CLEAR SENTENCES_1\n")
    sentences_1 = no_text_punkt(sentences_1)
    sentences_1 = clear_sentences(sentences_1)
    print("SENTENCES_1", *sentences[0:10], '\n')
    
    print("\nWORD2VEC ALGORITHM\n")
    model = gensim.models.word2vec.Word2Vec(sentences_1, size=vector_size, alpha = 0.025, 
                                   window = 3, min_count = 1, max_vocab_size=None, 
                                   seed = 42, workers = 8, iter = 100, sorted_vocab=1)
    model_path = name_dump+"/"+"model_"+str(max_len)
    model.save(model_path)
    
    time.sleep(0.1)
    print("CREATE PALETTE\n")
    word2index, index2word, eos, bos = create_palette(fd, probabilities[1], model, max_len, name_dump)
    
    time.sleep(0.1)
    print("CREATE DATA AND TARGET\n")
    gen_data = create_data(word2index, sentences, batch_size=128)
    train_data, train_target = next(gen_data)
    
    print("TRAIN DATA", train_data[0:10])
    print("TRAIN_LABELS", train_target[0:10])
    
    print("\FINISHED")
    #return whole_text, sentences, sentences_1, fd, probabilities, model, pallete, eos, bos, train_data, train_target
    return gen_data, word2index, index2word, model, max_len, sentences, gen_data
if __name__ == "__main__":
    #text, sents, sents_2, fd, probs, model, palette, eos, bos, train_data, train_target = execute_main()
    products = execute_main()
    gen_data, word2index, index2word, model, max_len, sentences, gen_data = products