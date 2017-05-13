#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 17:58:23 2017

@author: kinshiryuu-burp
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 20:25:04 2017

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

from multiprocessing import Process, Manager

import nltk
from nltk import FreqDist
from nltk import pos_tag
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

#ner_tagger = StanfordNERTagger('/home/kinshiryuu-burp/PROJEKTS/NATURAL_LANGUAGE_PROCESSING/stanford-#ner-2016-10-31/classifiers/english.all.3class.distsim.crf.ser.gz', 
#                             '/home/kinshiryuu-burp/PROJEKTS/NATURAL_LANGUAGE_PROCESSING/stanford-#ner-2016-10-31/stanford-ner.jar')

vector_size = 300
length = 30

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

"""def extract_entities(text):
    entities = []
    for sent in text:
        sent_ent = []
        for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
            if hasattr(chunk, 'node'):
                print(chunk.node, " ".join(c[0] for c in chunk.leaves()))
                sent_ent.append((chunk.node, (c[0] for c in chunk.leaves())))
        entities.append(sent_ent)
    
    return entities
"""


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
                line.append('EOS')
                whole_text += line
    
    return whole_text, sentences

def max_length(sentences):
    max_len = 0
    for sent in sentences:
        if len(word_tokenize(sent)) > max_len:
            max_len = len(word_tokenize(sent))
                
        #if len(sent) == 915:
        #    print(sent)
    max_len += 1
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
        clear_sent = []
        for word in sent:
            if word not in string.punctuation:
                clear_sent.append(word.lower())
        clear_sent.append('eos')
        if len(clear_sent) != 0:
            new_sents.append(clear_sent)
    
    return new_sents
        

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

def prob_dist(word_list):
    fd = FreqDist(word_list)
    funcs = [ELEProbDist, MLEProbDist, LaplaceProbDist, SimpleGoodTuringProbDist]
    probabilities = []
    for i, func in enumerate(funcs):
        print(i)
        probabilities.append(func(fd))
    
    return fd, probabilities


####################


def clear_sentences(sentences):
    clear_sents = []
    for i, sent in enumerate(sentences):
        sentences[i] = clear_text(sentences[i])
        if len(sentences[i]) != 0:
            clear_sents.append(sentences[i])
            
    return clear_sents

def padding_sents(sentences, max_len, name_dump=None):
    #print("\nPADDING STARTED\n\n")
    
    for i in range(len(sentences)):
        while len(sentences[i]) < max_len:
            sentences[i].append('eos')
            
    #print("\nPADDING FINISHED\n")
    if name_dump:
        print("SAVING SENTENCES\n")
        sents_path = name_dump+"/"+"sentences_"+str(max_len)+".pkl"
        with open(sents_path, "wb") as file:
            pickle.dump(sentences, file)
	    
        print("SENTENCES SAVED\n")
####################

def create_palette(fd, prob, model, max_len, name_dump):    
    word2index = dict()
    index2word = dict()
    items = sorted(list(fd.keys()))
    threshold = 0.75
    for i, key in enumerate(items):
        if key == 'eos':
            word2index[key] = (list(model.wv.word_vec(key)), 0.)
            index2word[str(list(model.wv.word_vec(key)))] = (key, 0.)
        elif pos_tag([key])[0][1] in ['NN', 'NNP', 'NNS', 'NNPS', 'PRP', 'PRP$', 'SYM']:
            proba = (2-prob.prob(key))/2
            if proba >= threshold:
                word2index[key] = (list(model.wv.word_vec(key)), 1.)
                index2word[str(list(model.wv.word_vec(key)))] = (key, 1.)
            else:
                word2index[key] = (list(model.wv.word_vec(key)), 0.)
                index2word[str(list(model.wv.word_vec(key)))] = (key, 0.)
                
        elif pos_tag([key])[0][1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'POS']:
            proba = (1.5-prob.prob(key))/2
            if proba >= threshold:
                word2index[key] = (list(model.wv.word_vec(key)), 1.)
                index2word[str(list(model.wv.word_vec(key)))] = (key, 1.)
            else:
                word2index[key] = (list(model.wv.word_vec(key)), 0.)
                index2word[str(list(model.wv.word_vec(key)))] = (key, 0.)
                
        elif pos_tag([key])[0][1] in ['CC', 'CD', 'DT', 'IN', 'LS']:
            proba = (1.-prob.prob(key))/2
            if proba >= threshold:
                word2index[key] = (list(model.wv.word_vec(key)), 1.)
                index2word[str(list(model.wv.word_vec(key)))] = (key, 1.)
            else:
                word2index[key] = (list(model.wv.word_vec(key)), 0.)
                index2word[str(list(model.wv.word_vec(key)))] = (key, 0.)
                
        elif pos_tag([key])[0][1] in ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']:
            proba = (1.5-prob.prob(key))/2
            if proba >= threshold:
                word2index[key] = (list(model.wv.word_vec(key)), 1.)
                index2word[str(list(model.wv.word_vec(key)))] = (key, 1.)
            else:
                word2index[key] = (list(model.wv.word_vec(key)), 0.)
                index2word[str(list(model.wv.word_vec(key)))] = (key, 0.)
                
        elif pos_tag([key])[0][1] in ['JJ', 'JJR', 'JJS']:
            proba = (1.5-prob.prob(key))/2
            if proba >= threshold:
                word2index[key] = (list(model.wv.word_vec(key)), 1.)
                index2word[str(list(model.wv.word_vec(key)))] = (key, 1.)
            else:
                word2index[key] = (list(model.wv.word_vec(key)), 0.)
                index2word[str(list(model.wv.word_vec(key)))] = (key, 0.)
                
        else:
            proba = 1.-prob.prob(key)
            if proba >= threshold:
                word2index[key] = (list(model.wv.word_vec(key)), 1.)
                index2word[str(list(model.wv.word_vec(key)))] = (key, 1.)
            else:
                word2index[key] = (list(model.wv.word_vec(key)), 0.)
                index2word[str(list(model.wv.word_vec(key)))] = (key, 0.)
    
    eos = word2index['eos']
    word2index_path = name_dump+"/"+"word2index_"+str(max_len)+".pkl"
    index2word_path = name_dump+"/"+"index2word_"+str(max_len)+".pkl"
    name_eos = name_dump+"/"+"eos_"+str(max_len)+".pkl"
    
    print("\nSAVING WORD2INDEX, INDEX2WORD AND EOS\n")
    time.sleep(1)
    with open(word2index_path, "wb") as fp:
        pickle.dump(word2index, fp)
        
    with open(index2word_path, "wb") as fb:
        pickle.dump(index2word, fb)
    
    with open(name_eos, "wb") as fe:
        pickle.dump(eos, fe)        
    print("\nSAVING FINISHED\n")
    time.sleep(1)
    return word2index, index2word, eos


def data_generator(word2index, sentences, batch_size = 128):
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
            train_sent = []
            label_sent = []
            for word in sent:
                train_sent.append(word2index[word][0])
                label_sent.append(word2index[word][1])
            train_sample.append(train_sent)
            train_label.append(label_sent)
        
        train_sample = np.array(train_sample)
        #train_sample = train_sample.astype(np.float64)
        train_label = np.array(train_label, dtype = np.int32)
        #print(train_sample)
        #print(train_label)
        yield train_sample, train_label
        
def sent_to_idx(word2index, sentence):
    train_sent = []
    for word in sentence:
        train_sent.append(word2index[word][0])
    train_sample = np.array(train_sent)
    return train_sample

def read_results(example, probs, index2word):
    sentences = []
    for i, sent in enumerate(example):
        senta = []
        check = []
        for j, word in enumerate(sent):
            check.append(index2word[str(list(word))][0])
            if probs[i][j] == 1.:
                senta.append(index2word[str(list(word))][0])
        print("Example:", check)
        print("Result:", senta)
        time.sleep(1)
        sentences.append(senta)
    
    return sentences
    

def random_example_test(train_data, pallete):
    idx = np.random.randint(train_data.shape[0])
    example = train_data[idx]
    for i in range(example.shape[0]):
        for j in range(len(pallete)):
            compare = np.array(pallete[j][1])
            if (example[i] == compare).all():
                print(pallete[j][0])
                
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
                
def multipr_func(sentences):
    div = 8
    train_data = list()
    train_target = list()
    with Manager() as manager:
        training_data = manager.list()
        training_target = manager.list()
        palte = manager.dict(word2index)
        rrange = int(np.ceil(len(sentences)/div))
        processes = list()
        for i in range(div):
            print("Iteration",(i+1))
            #numbers = range((i*rrange), ((i+1)*rrange))
            elems = sentences[(i*rrange):(i+1)*rrange]
            #print(elems)
            processes.append(Process(target=data_generator, args=(training_data, training_target, elems, palte)))
            
        print("PRS STARTED")
        for p in processes:
            p.start()
        print("PRS FINISHING\n")
        for p in processes:
            p.join()
            print("FINITO")
        print("\nPRS FINISHED\n")
            
            
        for i in range(len(training_data)):
            train_data.append(training_data[i])
            train_target.append(training_target[i])
            
        train_data = np.array(train_data)
        train_target = np.array(train_target)
        train_target = np.reshape(train_target, (train_target.shape[0], train_target.shape[1],1))
        
    return train_data, train_target

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
    name_dump = "./PROJECT_DUMP/POS_example[length_"+str(max_len)+"_vector_" + str(vector_size)+"]_|"+localtime+"|_"
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
    
    del sentences_1
    print("CREATE PALETTE\n")
    word2index, index2word, eos = create_palette(fd, probabilities[0], model, max_len, name_dump)
    #print("PALETTE", pallete[0:10])
    
    print("CREATE DATA AND TARGET\n")
    time.sleep(0.1)
    generate_data = data_generator(word2index, sentences, batch_size=128)
    print("\nCREATE DATA AND TARGET - FINISHED\n")   
    #print("TRAIN DATA", train_data[0:10])
    #print("TRAIN_LABELS", train_target[0:10])
    
    print("\nTESTING\n")
    """for i in range(10):
        print("\n{} EXAMPLE".format(i+1))
        random_example_test(train_data, pallete)
        time.sleep(1.5)
        print('\n')
    """
    
    print("FINISHED!!")
    return generate_data, word2index, index2word, model, max_len, sentences

    
    

if __name__ == "__main__":
    products = execute_main()
    generate_data, word2index, index2word, model, max_len, sentences = products
