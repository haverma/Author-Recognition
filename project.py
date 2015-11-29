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

'''def parse_input(train_file):
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
    return list_excerpts_label, list_excerpts, word_dictionary'''


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


            






if __name__=="__main__":
    #data = parse_input("data/project_articles_train")
    #get_positive_examples(data, "train_positive")
    topic_words = load_topic_words("topic_words.ts", 1000)
    fp = open("topic_top_1000.txt","w+")
    for word in topic_words:
        fp.write(word + "\n")
    fp.close()
