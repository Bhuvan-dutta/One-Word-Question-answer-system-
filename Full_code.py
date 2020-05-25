# -*- coding: utf-8 -*-
"""
Created on Tue May 26 00:35:01 2020

@author: Bhuvan Dutta
"""


import nltk
import gensim
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import io
import operator
import re
import spacy

########################Declaring variables############################################ 

nlp = spacy.load("en_core_web_sm")
new_line='\n'
paragraph_separator='. '
lines_separator='\.+\D'
One=1
Zero=0
file_mode="rb"
#'rb'
encoding='utf-8'
exact_line_result=[]
cosine_result=[]
entity_dictionary={}
language='english'
boolean_True=True
Set=set()
sqrt_value=0.5

sw = stopwords.words(language)  

################################Required Input from users##################################

User_Question ="How much Growth was led by UK?"
#User_Question ="Who recognized TCS as Customer Success Partner of the Year?"
#User_Question="Why TCS Customers value the lower attrition?"
#User_Question="What is the revenue growth for Life Sciences & Healthcare in TCS?"
pdf_path=r'C:\Users\bhuva_pxpvpbh\OneDrive\Desktop\Press Release - INR.pdf'

#######################Creating mask and their entity mappings###################### 
mask_list=["What","Who","When","How many","How much","Why","How did"]
mask_direct_dictionary={'What':['PRODUCT','EVENT','LAW','LANGUAGE','CARDINAL','MONEY','PERCENT','WORK_OF_ART'],
                        'Who':['PERSON','NOORP','ORG','EVENT','GPE'],
                        'Where':['GPE','LOC','FAC'],
                        'When':['TEMPORAL','DATE','TIME'],
                        'How much':['NUMERIC','PERCENT','CARDINAL','QUANTITY','MONEY'],
                        'How many':['NUMERIC','PERCENT','CARDINAL','QUANTITY','MONEY']}

mask_explanation_dictionary={'Why':['ADP','SCONJ','VERB'],'How did':['ADP','VERB']}
reason_joiner=['for','because','due to']
#mask_explanation_dictionary={'Why':['ADP','SCONJ','VERB']}

#######################Function for extracting raw text data from PDF################
def pdf_to_text_conversion(path):
    
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr,laparams=laparams)
    fp = open(path,file_mode)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    maxpages = Zero
    caching = boolean_True
    pagenos = Set
    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages,caching=caching,check_extractable=boolean_True):
        interpreter.process_page(page)
    fp.close()
    device.close()
    text = retstr.getvalue()
    retstr.close()
    
    return text

#################stripping the sentences from raw text and splitting paragraph wise##############

Lines=pdf_to_text_conversion(pdf_path)

#################Removing blank strings('') from list of sentences###################################

file_docs = []
tokens1 = sent_tokenize(Lines)
for line in tokens1:
    file_docs.append(line)
print(len(file_docs))

print("Number of documents:",len(file_docs))

gen_docs = [[w.lower() for w in word_tokenize(text)] for text in file_docs]

dictionary = gensim.corpora.Dictionary(gen_docs)
print(dictionary.token2id)

corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

tfidf = gensim.models.TfidfModel(corpus)
for doc in tfidf[corpus]:
    print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])
    
sims = gensim.similarities.Similarity(r'C:\Users\bhuva_pxpvpbh\OneDrive\Desktop',tfidf[corpus],num_features=len(dictionary))

file2_docs=[]
#with open ('demofile2.txt') as f:
tokens2 = sent_tokenize(User_Question)
for line in tokens2:
    file2_docs.append(line)

print("Number of documents:",len(file2_docs))  
for line in file2_docs:
    query_doc = [w.lower() for w in word_tokenize(line)]
    query_doc_bow = dictionary.doc2bow(query_doc) #upda
    
query_doc_tf_idf = tfidf[query_doc_bow]

print(type(sims[query_doc_tf_idf]))


max_index, max_value = max(enumerate(sims[query_doc_tf_idf]), key=operator.itemgetter(One))

#####from here need to write#######################

RelevantContext=file_docs[max_index]
parapgraph_lines=sent_tokenize(RelevantContext)
parapgraph_lines = list(filter(None, parapgraph_lines))

if len(parapgraph_lines)>1:
    para_docs = []
    #para_tokens = sent_tokenize(RelevantContext)
    for line in parapgraph_lines:
        para_docs.append(line)
    print("1-Number of documents:",len(para_docs))
    
    gen_docs = [[w.lower() for w in word_tokenize(text)] for text in para_docs]
    
    dictionary = gensim.corpora.Dictionary(gen_docs)
    print("dictionary.token2id:",dictionary.token2id)
    
    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
    
    tfidf = gensim.models.TfidfModel(corpus)
    for doc in tfidf[corpus]:
        print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])
    
    sims = gensim.similarities.Similarity(r'C:\Users\bhuva_pxpvpbh\OneDrive\Desktop',tfidf[corpus],num_features=len(dictionary))
    
    para_file2_docs=[]
    #with open ('demofile2.txt') as f:
    para_tokens2 = sent_tokenize(User_Question)
    for line in para_tokens2:
        para_file2_docs.append(line)
    
    print("Number of documents:",len(file2_docs))  
    for line in para_file2_docs:
        query_doc = [w.lower() for w in word_tokenize(line)]
        query_doc_bow = dictionary.doc2bow(query_doc) #upda
        
    query_doc_tf_idf = tfidf[query_doc_bow]

    print(type(sims[query_doc_tf_idf]))
    
    
    max_index, max_value = max(enumerate(sims[query_doc_tf_idf]), key=operator.itemgetter(One))

    doc = nlp(parapgraph_lines[max_index])
else:
    doc = nlp(RelevantContext)
    
entity_explanation_dictionary={}

for ent in doc.ents:
    entity_dictionary[ent.text]=ent.label_ 
for token in doc:
    entity_explanation_dictionary[token.text]=token.pos_
    

mask = ''.join(list(filter(User_Question.startswith, mask_list)))
if mask in mask_direct_dictionary:
    main_objective_to_find=mask_direct_dictionary.get(mask)
    for name,entity in entity_dictionary.items():
        if entity in main_objective_to_find:
            Named_entity_list = [key  for (key, value) in entity_dictionary.items() if value == entity]
    
            word_docs = [[w.lower() for w in word_tokenize(text)] for text in Named_entity_list]
    
            dictionary = gensim.corpora.Dictionary(word_docs)
            corpus = [dictionary.doc2bow(word_doc) for word_doc in word_docs]
            tfidf = gensim.models.TfidfModel(corpus)
            for doc in tfidf[corpus]:
                print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])
            
            sims = gensim.similarities.Similarity(r'C:\Users\bhuva_pxpvpbh\OneDrive\Desktop',tfidf[corpus],num_features=len(dictionary))
            
            word_file_docs=[]
            word_tokens = sent_tokenize(User_Question)
            for line in word_tokens:
                word_file_docs.append(line)
        
            for line in word_file_docs:
                word_doc = [w.lower() for w in word_tokenize(line)]
                word_doc_bow = dictionary.doc2bow(word_doc)
            word_doc_tf_idf = tfidf[word_doc_bow]
    
            exact_index, word_value = max(enumerate(sims[word_doc_tf_idf]), key=operator.itemgetter(One))        
            print("Exact answer is:",Named_entity_list[exact_index])
            break
else:
    main_objective_to_find=mask_explanation_dictionary.get(mask)
    for name,entity in entity_explanation_dictionary.items():
        if entity in main_objective_to_find:
            Named_entity_list = [key  for (key, value) in entity_explanation_dictionary.items() if value == entity]
    #for lines in parapgraph_lines:
    for entities in Named_entity_list:
        if entities in reason_joiner:
            required_answer=entities+RelevantContext.split(entities)[-1]
            print("Exact answer is::",required_answer)
            break
      
            





