from nltk import sent_tokenize, word_tokenize
import pandas as pd
import numpy as np
import nltk
import os
from nltk import ne_chunk
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tree import Tree
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

os.getcwd()
os.chdir('C:\\Users\\Sharath\\Desktop\\senttext')

df=open('tesla.txt',encoding='utf8').read()
stopwords=set(stopwords.words('english'))
dt=sent_tokenize(df)

filt_df=[w for w in df_word if not w in stopwords]
df_txt=' '.join(filt_df)
df_sent=sent_tokenize(df)

list1=[]
list2=[]
list3=[]
list4=[]
list5=[]

for i in df_sent:
    if 'competitor' in i:
        list1.append(i)
    if 'peer' in i:
        list2.append(i)
    if 'customer' or 'client' in i:
        list3.append(i)
    if 'partner' in i:
        list4.append(i)
    if 'supplier' in i:
        list5.append(i)
        
sampcomp = ' '.join(list1)
samppeer = ' '.join(list2)
sampcust = ' '.join(list3)
samppart = ' '.join(list4)
sampsup = ' '.join(list5)

custom_sent_tokenizer = PunktSentenceTokenizer(df_txt)

tokcomp = custom_sent_tokenizer.tokenize(sampcomp)
tokpeer = custom_sent_tokenizer.tokenize(samppeer)
tokcust = custom_sent_tokenizer.tokenize(sampcust)
tokpart = custom_sent_tokenizer.tokenize(samppart)
toksup = custom_sent_tokenizer.tokenize(sampsup)

tokenized_sentences_comp = [nltk.word_tokenize(sentence) for sentence in tokcomp]
tagged_sentences_comp = [nltk.pos_tag(sentence) for sentence in tokenized_sentences_comp]
chunked_sentences_comp = nltk.ne_chunk_sents(tagged_sentences_comp, binary=True)

tokenized_sentences_peer = [nltk.word_tokenize(sentence) for sentence in tokpeer]
tagged_sentences_peer = [nltk.pos_tag(sentence) for sentence in tokenized_sentences_peer]
chunked_sentences_peer = nltk.ne_chunk_sents(tagged_sentences_peer, binary=True)

tokenized_sentences_cust = [nltk.word_tokenize(sentence) for sentence in tokcust]
tagged_sentences_cust = [nltk.pos_tag(sentence) for sentence in tokenized_sentences_cust]
chunked_sentences_cust = nltk.ne_chunk_sents(tagged_sentences_cust, binary=True)

tokenized_sentences_part = [nltk.word_tokenize(sentence) for sentence in tokpart]
tagged_sentences_part = [nltk.pos_tag(sentence) for sentence in tokenized_sentences_part]
chunked_sentences_part = nltk.ne_chunk_sents(tagged_sentences_part, binary=True)

tokenized_sentences_sup = [nltk.word_tokenize(sentence) for sentence in toksup]
tagged_sentences_sup = [nltk.pos_tag(sentence) for sentence in tokenized_sentences_sup]
chunked_sentences_sup = nltk.ne_chunk_sents(tagged_sentences_sup, binary=True)


def extract_entity_names(t):
    x = []

    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            x.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                x.extend(extract_entity_names(child))

    return x

competitor = []
peer = []
customer = []
partner = []
supplier = []



gen_list=[chunked_sentences_comp,chunked_sentences_peer,chunked_sentences_cust,chunked_sentences_part,chunked_sentences_sup]
for tree in gen_list[0]:    
    # Print results per sentence
    # print extract_entity_names(tree)
    competitor.extend(extract_entity_names(tree))
for tree in gen_list[1]:    
    # Print results per sentence
    # print extract_entity_names(tree)
    peer.extend(extract_entity_names(tree))
for tree in gen_list[2]:    
    # Print results per sentence
    # print extract_entity_names(tree)
    if len(tree)<100:
        customer.extend(extract_entity_names(tree))
for tree in gen_list[3]:    
    # Print results per sentence
    # print extract_entity_names(tree)
    partner.extend(extract_entity_names(tree))
for tree in gen_list[4]:    
    # Print results per sentence
    # print extract_entity_names(tree)
    supplier.extend(extract_entity_names(tree))
    
competitors=set(competitor)
peers = set(peer)
customers = set(customer)
partners = set(partner)
suppliers = set(supplier)
