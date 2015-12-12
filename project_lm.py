from __future__ import division
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
import math
import subprocess
import os
import csv
import pdb
import operator
import re
from sets import Set
from collections import OrderedDict, Counter
import linecache
from svmutil import *
import random
import nltk






def ngram_tokenize(n, excerpt):
    tuplearray=[]
    sentences = sent_tokenize(excerpt)
    #sentences[0] = "<s>" + sentences[0]
    parsedarray = []
    for i in range(len(sentences)):
        word_list = word_tokenize(sentences[i])
        word_list.insert(0,"<s>")
        word_list.append("</s>")
        parsedarray += word_list
    for i in range(0, len(parsedarray)-n + 1):
        newlist = parsedarray[i:i+n]
        tuplearray.append(tuple(newlist))
    return tuplearray

def ngram_pos_tokenize(n, list_pos):
    tuplearray=[]
    for i in range(0, len(list_pos)-n + 1):
        newlist = list_pos[i:i+n]
        tuplearray.append(tuple(newlist))
    return tuplearray




def flatten(listoflists):
    flat_list=[]
    for inner_list in listoflists:
        flat_list.extend(inner_list)
    return flat_list


def get_all_files(directory):
    list_entries=os.listdir(directory)
    list_files=[entry for entry in list_entries if os.path.isfile(os.path.join(directory,entry))]
    return list_files

def get_fullpath_files(directory):
    list_entries=os.listdir(directory)
    list_files=[os.path.join(directory,entry) for entry in list_entries if os.path.isfile(os.path.join(directory,entry))]
    return list_files

def parse_input_mft(train_file):
    fp = open(train_file,"r")
    list_excerpts_label = []
    for line in fp:
        tokens = line.rsplit('\t',1)
        list_excerpts_label.append((tokens[0], int(tokens[1].strip())))
    fp.close()
    list_excerpts = [tuple_entry[0] for tuple_entry in list_excerpts_label]
    excerpts_list_words = []
    for excerpt in list_excerpts:
        excerpt.strip()
        word_list = word_tokenize(excerpt.decode('utf8'))
        excerpts_list_words.append(word_list)
    excerpts_list_words = flatten(excerpts_list_words)
    word_dictionary = defaultdict(int)
    for token in excerpts_list_words:
        word_dictionary[token]+=1
    counter_words = Counter(word_dictionary)
    list_words = []
    for k,v in counter_words.most_common(3000):
        list_words.append(k)
    return list_excerpts_label, list_excerpts, list_words

def get_stopwords(filename):
    list_stopwords = []
    fp = open(filename,"r")
    for line in fp:
        list_stopwords.append(line.strip())
    return list_stopwords


def parse_trigram_mft(train_file):
    fp = open(train_file,"r")
    list_excerpts_label = []
    for line in fp:
        tokens = line.rsplit('\t',1)
        list_excerpts_label.append((tokens[0], int(tokens[1].strip())))
    fp.close()
    list_excerpts = [tuple_entry[0] for tuple_entry in list_excerpts_label]
    excerpts_list_trigams = []
    for excerpt in list_excerpts:
        excerpt.strip()
        ngrams_list = ngram_tokenize(3, excerpt.decode('utf8'))
        excerpts_list_trigams.append(ngrams_list)
    excerpts_list_trigams = flatten(excerpts_list_trigams)
    trigram_dictionary = defaultdict(int)
    for token in excerpts_list_trigams:
        trigram_dictionary[token]+=1
    counter_trigrams = Counter(trigram_dictionary)
    list_trigrams = []
    for k,v in counter_trigrams.most_common(5000):
        list_trigrams.append(k)
    return list_excerpts_label, list_excerpts, list_trigrams

def parse_bigram_mft(train_file):
    stop_tokens = (u'</s>', u'<s>')
    fp = open(train_file,"r")
    list_excerpts_label = []
    for line in fp:
        tokens = line.rsplit('\t',1)
        list_excerpts_label.append((tokens[0], int(tokens[1].strip())))
    fp.close()
    list_excerpts = [tuple_entry[0] for tuple_entry in list_excerpts_label]
    excerpts_list_bigams = []
    for excerpt in list_excerpts:
        excerpt.strip()
        ngrams_list = ngram_tokenize(2, excerpt.decode('utf8'))
        excerpts_list_bigams.append(ngrams_list)
    excerpts_list_bigams = flatten(excerpts_list_bigams)
    bigram_dictionary = defaultdict(int)
    for token in excerpts_list_bigams:
        if(token != stop_tokens):
            bigram_dictionary[token]+=1
    counter_bigrams = Counter(bigram_dictionary)
    list_bigrams = []
    for k,v in counter_bigrams.most_common(5000):
        list_bigrams.append(k)
    return list_excerpts_label, list_excerpts, list_bigrams


def parse_quads_mft(train_file):
    fp = open(train_file,"r")
    list_excerpts_label = []
    for line in fp:
        tokens = line.rsplit('\t',1)
        list_excerpts_label.append((tokens[0], int(tokens[1].strip())))
    fp.close()
    list_excerpts = [tuple_entry[0] for tuple_entry in list_excerpts_label]
    excerpts_list_trigams = []
    for excerpt in list_excerpts:
        excerpt.strip()
        ngrams_list = ngram_tokenize(4, excerpt.decode('utf8'))
        excerpts_list_trigams.append(ngrams_list)
    excerpts_list_trigams = flatten(excerpts_list_trigams)
    trigram_dictionary = defaultdict(int)
    for token in excerpts_list_trigams:
        trigram_dictionary[token]+=1
    counter_trigrams = Counter(trigram_dictionary)
    list_trigrams = []
    for k,v in counter_trigrams.most_common(500):
        list_trigrams.append(k)
    return list_excerpts_label, list_excerpts, list_trigrams


def parse_pentas_mft(train_file):
    fp = open(train_file,"r")
    list_excerpts_label = []
    for line in fp:
        tokens = line.rsplit('\t',1)
        list_excerpts_label.append((tokens[0], int(tokens[1].strip())))
    fp.close()
    list_excerpts = [tuple_entry[0] for tuple_entry in list_excerpts_label]
    excerpts_list_trigams = []
    for excerpt in list_excerpts:
        excerpt.strip()
        ngrams_list = ngram_tokenize(5, excerpt.decode('utf8'))
        excerpts_list_trigams.append(ngrams_list)
    excerpts_list_trigams = flatten(excerpts_list_trigams)
    trigram_dictionary = defaultdict(int)
    for token in excerpts_list_trigams:
        trigram_dictionary[token]+=1
    counter_trigrams = Counter(trigram_dictionary)
    list_trigrams = []
    for k,v in counter_trigrams.most_common(600):
        list_trigrams.append(k)
    return list_excerpts_label, list_excerpts, list_trigrams


def parse_input(train_file):
    fp = open(train_file,"r")
    list_excerpts_label = []
    for line in fp:
        tokens = line.rsplit('\t',2)
        list_pos = [nltk.tag.str2tuple(t)[1] for t in tokens[0].split()]
        list_excerpts_label.append((tokens[1], int(tokens[2].strip()), list_pos))
    fp.close()
    return list_excerpts_label

def parse_test_input(testfile):
    fp = open(testfile,"r")
    list_excerpts_label = []
    for line in fp:
        list_excerpts_label.append((line, 0))
    fp.close()
    return list_excerpts_label


def get_bigram_dict(list_bigrams):
    vocab_index = {}
    for i in range(1,(len(list_bigrams) + 1)):
        vocab_index[list_bigrams[i-1]] =  8000 + i
    return vocab_index

def get_trigram_dict(list_trigrams):
    vocab_index = {}
    for i in range(1,(len(list_trigrams) + 1)):
        vocab_index[list_trigrams[i-1]] =  3000 + i
    return vocab_index

def get_quad_dict(list_quads):
    vocab_index = {}
    for i in range(1,(len(list_quads) + 1)):
        vocab_index[list_quads[i-1]] =  13044 + i
    return vocab_index

def get_penta_dict(list_quads):
    vocab_index = {}
    for i in range(1,(len(list_quads) + 1)):
        vocab_index[list_quads[i-1]] =  13545 + i
    return vocab_index



def get_vocab_dict(list_words):
    vocab_list = sorted(list_words)
    vocab_index = {}
    for i in range(1,(len(vocab_list) + 1)):
        vocab_index[vocab_list[i-1]] = i
    return vocab_index


def generate_count_dict(excerpt,vocab_dict):
    excerpt_count_dict = defaultdict(int)
    words_excerpt = word_tokenize(excerpt.decode('utf8'))
    for word in words_excerpt:
        excerpt_count_dict[word]+=1
    dict_index_count = {}
    for word in excerpt_count_dict:
        if word in vocab_dict:
            dict_index_count[vocab_dict[word]] = excerpt_count_dict[word]
    return dict_index_count


def generate_trigram_dict(excerpt, trigram_dict):
    excerpt_count_dict = defaultdict(int)
    trigrams_excerpt = ngram_tokenize(3, excerpt.decode('utf8'))
    for trigram in trigrams_excerpt:
        excerpt_count_dict[trigram]+=1
    dict_index_count = {}
    for trigram in excerpt_count_dict:
        if trigram in trigram_dict:
            dict_index_count[trigram_dict[trigram]] = excerpt_count_dict[trigram]
    return dict_index_count


def generate_quadgram_dict(excerpt, quaddict):
    excerpt_count_dict = defaultdict(int)
    trigrams_excerpt = ngram_tokenize(4, excerpt.decode('utf8'))
    for trigram in trigrams_excerpt:
        excerpt_count_dict[trigram]+=1
    dict_index_count = {}
    for trigram in excerpt_count_dict:
        if trigram in quaddict:
            dict_index_count[quaddict[trigram]] = excerpt_count_dict[trigram]
    return dict_index_count

def generate_pentgram_dict(excerpt, quaddict):
    excerpt_count_dict = defaultdict(int)
    trigrams_excerpt = ngram_tokenize(5, excerpt.decode('utf8'))
    for trigram in trigrams_excerpt:
        excerpt_count_dict[trigram]+=1
    dict_index_count = {}
    for trigram in excerpt_count_dict:
        if trigram in quaddict:
            dict_index_count[quaddict[trigram]] = excerpt_count_dict[trigram]
    return dict_index_count

def generate_bigram_dict(excerpt, bigram_dict):
    excerpt_count_dict = defaultdict(int)
    bigrams_excerpt = ngram_tokenize(2, excerpt.decode('utf8'))
    for bigram in bigrams_excerpt:
        excerpt_count_dict[bigram]+=1
    dict_index_count = {}
    for bigram in excerpt_count_dict:
        if bigram in bigram_dict:
            dict_index_count[bigram_dict[bigram]] = excerpt_count_dict[bigram]
    return dict_index_count


def train_test_model(train_datafile, test_datafile):
        y,x = svm_read_problem(train_datafile)
        test_y, test_x = svm_read_problem(test_datafile)
        model_train = svm_train(y,x, '-t 0 -e .01 -m 1000 -h 0')
        p_labs, p_acc, p_vals = svm_predict(test_y, test_x, model_train)
        return p_labs, p_acc, p_vals

def train_test_model_final(train_datafile, test_datafile):
        y,x = svm_read_problem(train_datafile)
        test_y, test_x = svm_read_problem(test_datafile)
        model_train = svm_train(y,x, '-t 0 -e .01 -m 1000 -h 0')
        p_labs, p_acc, p_vals = svm_predict([0]*len(test_x), test_x, model_train)
        return p_labs



def train_crossvalidation(train_datafile, testlinenos):
    train_lines = set(range(1,12114))
    train_lines.difference_update(testlinenos)
    fp_test = open("svm_test.txt","w+")
    fp_train = open("svm_train.txt","w+")
    for line in testlinenos:
	cur_line = linecache.getline(train_datafile, line)
        fp_test.write(cur_line)
    for line in train_lines:
        cur_line = linecache.getline(train_datafile, line)
 	fp_train.write(cur_line)
    fp_train.close()
    fp_test.close()
    p_labs, p_acc, p_vals = train_test_model("svm_train.txt","svm_test.txt")
    return p_acc

def get_pos_dict(pos_list):
    flat_pos_list = flatten(pos_list)
    pos_dict = defaultdict(int)
    for token in flat_pos_list:
        pos_dict[token]+=1
    counter_pos = Counter(pos_dict)
    list_pos = []
    for k,v in counter_pos.most_common(500):
        list_pos.append(k)
    vocab_index = {}
    for i in range(1,(len(list_pos) + 1)):
        vocab_index[list_pos[i-1]] =  13000 + i
    return vocab_index


def generate_pos_count_dict(list_pos, pos_dict):
    excerpt_count_dict = defaultdict(int)
    for pos in list_pos:
        excerpt_count_dict[pos]+=1
    dict_index_count = {}
    for pos in excerpt_count_dict:
        if pos in pos_dict:
            dict_index_count[pos_dict[pos]] = excerpt_count_dict[pos]
    return dict_index_count

def get_avaerage_sentence_length(excerpt):
    sentences = sent_tokenize(excerpt.decode('utf8'))
    words = word_tokenize(excerpt.decode('utf8'))
    average_length = len(words)/len(sentences)
    return average_length/20

def get_average_word_length(excerpt):
    words = word_tokenize(excerpt.decode('utf8'))
    total_length = 0
    for word in words:
        total_length+=len(word)
    return total_length/len(words)

def gettype_tokenratio(excerpt):
    words = word_tokenize(excerpt.decode('utf8'))
    dict_words = defaultdict(int)
    for word in words:
        dict_words[word]+=1
    return len(dict_words)/len(words)




def get_pos_ngrams_dict(pos_list):
    flat_pos_list = flatten(pos_list)
    bigram_cnt = Counter()
    trigram_cnt = Counter()
    bigram_list = ngram_pos_tokenize(2, flat_pos_list)
    trigram_list = ngram_pos_tokenize(3, flat_pos_list)
    for bigram in bigram_list:
        bigram_cnt[bigram]+=1
    for trigram in trigram_list:
        trigram_cnt[trigram]+=1
    bigrams = bigram_cnt.most_common(600)
    trigrams = trigram_cnt.most_common(700)
    bigram_index_count = {}
    trigram_index_count = {}
    for i in range(1,(len(bigrams) + 1)):
        bigram_index_count[bigrams[i-1][0]] =  14145 + i
    for i in range(1,(len(trigrams) + 1)):
        trigram_index_count[trigrams[i-1][0]] =  13055 + i
    return bigram_index_count


def get_trigram_pos_dict(pos_list, trigram_dict):
    excerpt_count_dict = defaultdict(int)
    trigrams_excerpt = ngram_pos_tokenize(3, pos_list)
    for trigram in trigrams_excerpt:
        excerpt_count_dict[trigram]+=1
    dict_index_count = {}
    for trigram in excerpt_count_dict:
        if trigram in trigram_dict:
            dict_index_count[trigram_dict[trigram]] = excerpt_count_dict[trigram]
    return dict_index_count

def get_bigram_pos_dict(pos_list, bigram_dict):
    excerpt_count_dict = defaultdict(int)
    bigrams_excerpt = ngram_pos_tokenize(2, pos_list)
    for bigram in bigrams_excerpt:
        excerpt_count_dict[bigram]+=1
    dict_index_count = {}
    for bigram in excerpt_count_dict:
        if bigram in bigram_dict:
            dict_index_count[bigram_dict[bigram]] = excerpt_count_dict[bigram]
    return dict_index_count



def generate_pos_tagged_file(inputfile, outfile):
    fp = open(inputfile,'r')
    fp_combined = open(outfile, 'w+')
    for line in fp:
        tokens = line.rsplit('\t',1)
        text = word_tokenize(tokens[0].strip().decode('utf8'))
        pos_tags = nltk.pos_tag(text)
        for entry in pos_tags:
            fp_combined.write(entry[0].encode('utf8') + "/" + entry[1].encode('utf8') + " ")
        fp_combined.write("\t")
        fp_combined.write(tokens[0].strip() + "\t")
        fp_combined.write(tokens[1].strip()+ "\n")
    fp_combined.close()
    fp.close()

        
    











if __name__=="__main__":
    data = parse_input("combined.txt")
    pos_list = [i[2] for i in data]
    pos_dict = get_pos_dict(pos_list)
    bigram_pos = get_pos_ngrams_dict(pos_list)
    list_dummy,list_dummy1,topic_words = parse_input_mft("data/project_articles_train")
    list_dummy, list_dummy1, most_freq_bigrams = parse_bigram_mft("data/project_articles_train")
    list_dummy, list_dummy1, most_freq = parse_trigram_mft("data/project_articles_train")
    list_dummy, list_dummy1, most_freq_quads = parse_quads_mft("data/project_articles_train")
    list_dummy, list_dummy1, most_freq_pentas = parse_pentas_mft("data/project_articles_train")

    vocab_dict = get_vocab_dict(topic_words)
    bigram_dict = get_bigram_dict(most_freq_bigrams)
    trigram_dict = get_trigram_dict(most_freq)
    quad_dict = get_quad_dict(most_freq_quads)
    pent_dict = get_penta_dict(most_freq_pentas)

    list_label_features = []
    for tuple_entry in data:
        list_label_features.append((tuple_entry[1], generate_count_dict(tuple_entry[0], vocab_dict), generate_trigram_dict(tuple_entry[0], trigram_dict), generate_bigram_dict(tuple_entry[0], bigram_dict), generate_pos_count_dict(tuple_entry[2], pos_dict),generate_quadgram_dict(tuple_entry[0], quad_dict), generate_pentgram_dict(tuple_entry[0], pent_dict),get_bigram_pos_dict(tuple_entry[2], bigram_pos)))
    svm_file = open("test_hit.txt", "w+")
    for instance in list_label_features:
        svm_file.write(str(instance[0]) + " ")
        od = OrderedDict(sorted(instance[1].items()))
        trigram_od = OrderedDict(sorted(instance[2].items()))
        bigram_od = OrderedDict(sorted(instance[3].items()))
        pos_unigram_od = OrderedDict(sorted(instance[4].items()))
        quad_od = OrderedDict(sorted(instance[5].items()))
        pent_od = OrderedDict(sorted(instance[6].items()))
        bigrampos_od = OrderedDict(sorted(instance[7].items()))
      
        for key in od:
            svm_file.write(str(key) + ":" + str(od[key])  + " ")
        for key in trigram_od:
            svm_file.write(str(key) + ":" + str(trigram_od[key])  + " ")
        for key in bigram_od:
            svm_file.write(str(key) + ":" + str(bigram_od[key])  + " ")
        for key in pos_unigram_od:
            svm_file.write(str(key) + ":" + str(pos_unigram_od[key])  + " ")
        for key in quad_od:
            svm_file.write(str(key) + ":" + str(quad_od[key])  + " ")
        for key in pent_od:
            svm_file.write(str(key) + ":" + str(pent_od[key])  + " ")
        for key in bigrampos_od:
            svm_file.write(str(key) + ":" + str(bigrampos_od[key])  + " ")
        
        svm_file.write("\n")
	# data_test = parse_input("data/project_articles_test")
    #get_positive_examples(data, "train_positive")
    #get_negative_examples(data, "train_negative")
    #print train_test_model("file_svm_train.txt", "file_svm_test.txt")
    # lines_tested = set()
    # left_set = set(range(1,12114))
    # total_acc = 0
    # counter = 0
    # while(True):
    #     if len(left_set) < 1000:
    #         counter+=1
    #         total_acc+=train_crossvalidation("test_hit.txt", left_set)[0]
    #         break
    #     else:
    #         counter+=1
    #         current_set = set(random.sample(list(left_set), 1000))
    #         lines_tested.union(current_set)
    #         left_set.difference_update(current_set)
    #         total_acc+=train_crossvalidation("test_hit.txt", current_set)[0]
    # print total_acc/counter
    #srilm_bigram_models("train_positive","/home1/h/hverma/Project_CIS539/cis530_project/language_models")
    #srilm_bigram_models("train_negative","/home1/h/hverma/Project_CIS539/cis530_project/language_models")
    # list_labels = train_test_model_final("11_12_train_word_length.txt","11_12_train_word_length_test.txt")
    # fp_result = open("testy.txt", "w+")
    # for label in list_labels:
    #     fp_result.write(str(int(label)) + "\n")
    # fp_result.close()









    
    
