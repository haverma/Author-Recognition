from __future__ import division
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
import math
import subprocess
import os
import csv
import operator
import re
from sets import Set
from collections import OrderedDict, Counter
import linecache
import random





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
    for k,v in counter_words.most_common(1000):
        list_words.append(k)
    return list_excerpts_label, list_excerpts, list_words


def parse_input(train_file):
    fp = open(train_file,"r")
    list_excerpts_label = []
    for line in fp:
        tokens = line.rsplit('\t',1)
        list_excerpts_label.append((tokens[0], int(tokens[1].strip())))
    fp.close()
    return list_excerpts_label

def get_positive_examples(list_excerpts_label,outfile):
    fp = open(outfile,"w+")
    for excerpt_label in list_excerpts_label:
        if(excerpt_label[1] == 1):
            fp.write(excerpt_label[0]+ "\n")
    fp.close()


def load_topic_words(topic_file, n):
    fp = open(topic_file,'r')
    list_tuples = []
    for line in fp:
        tokens = line.split(" ")
        if float(tokens[1].strip()) >= 10.0:
            list_tuples.append((tokens[0], float(tokens[1].strip())))
    list_tuples = sorted(list_tuples, key = lambda tuple_entry: tuple_entry[1], reverse=True)
    if n < len(list_tuples):
        tuple_topn = list_tuples[:n]
    else:
        tuple_topn = list_tuples
    list_words = [tuple_entry[0] for tuple_entry in tuple_topn]
    fp.close()
    return list_words

def get_vocab_dict(list_words):
    vocab_list = sorted(list_words)
    vocab_index = {}
    for i in range(1,(len(vocab_list) + 1)):
        vocab_index[vocab_list[i-1]] = i
    return vocab_index


def generate_test_dict(excerpt,vocab_dict):
    excerpt_count_dict = defaultdict(int)
    words_excerpt = word_tokenize(excerpt.decode('utf8'))
    for word in words_excerpt:
        excerpt_count_dict[word]+=1
    dict_index_count = {}
    for word in excerpt_count_dict:
        if word in vocab_dict:
            dict_index_count[vocab_dict[word]] =  excerpt_count_dict[word]/len(excerpt)
    return dict_index_count

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

def train_test_model(train_datafile, test_datafile):
        y,x = svm_read_problem(train_datafile)
        test_y, test_x = svm_read_problem(test_datafile)
        model_train = svm_train(y,x, '-t 0 -e .01 -m 1000 -h 0')
        p_labs, p_acc, p_vals = svm_predict(test_y, test_x, model_train)
        return p_labs, p_acc, p_vals



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
    print p_acc
	    
	    


if __name__=="__main__":
    data = parse_input("data/project_articles_train")
    #get_positive_examples(data, "train_positive")
    #topic_words = load_topic_words("topic_words.ts", 1000)
    list_dummy,list_dummy1,topic_words = parse_input_mft("data/project_articles_train")
    vocab_dict = get_vocab_dict(topic_words)
    list_label_features = []
    for tuple_entry in data:
        list_label_features.append((tuple_entry[1], generate_test_dict(tuple_entry[0], vocab_dict)))
    svm_file = open("file_svm_train_two.txt", "w+")
    for instance in list_label_features:
        svm_file.write(str(instance[0]) + " ")
        od = OrderedDict(sorted(instance[1].items()))
        for key in od:
            svm_file.write(str(key) + ":" + "{:.7f}".format(od[key])  + " ")
        svm_file.write("\n")
	#data_test = parse_input("data/project_articles_test")
    #get_positive_examples(data, "train_positive")
    #print train_test_model("file_svm_train.txt", "file_svm_test.txt")
    '''lines_tested = set()
    left_set = set(range(1,12114))
    total_acc = 0
    counter = 0
    while(True):
        if len(left_set) < 1000:
            counter+=1
        total_acc+=train_crossvalidation("file_svm_train_one.txt", left_set)[0]
        break
    else:
            counter+=1
        current_set = set(random.sample(list(left_set), 1000))
            lines_tested.union(current_set)
        left_set.difference_update(current_set)
        total_acc+=train_crossvalidation("file_svm_train_one.txt", current_set)[0]
    print total_acc/counter'''

	





    
    
