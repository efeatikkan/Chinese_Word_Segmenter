# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 00:19:02 2019

@author: Efe
"""
import pickle
import numpy as np
import os
import hanziconv

def spacer(inn):
  inn=inn.replace('  ', '\u3000').replace(' ','\u3000')
  return(inn)

def data_converter(inn):
    "Function to create input file without spaces"
    inn_file = inn.replace('\u3000', '').replace('\n', '')
    return(inn_file)
    
def beis_generator(inn) :
    """
    Takes input one string/ line. 
    Turns each character to BEIS encode according to past,current and future characters.
    """
    
    punc_list= ['●',"－","～","＇","｀","《","》","？","；","：","，","。","、","！","“","”","『","』","（","）","「","」","‘","’","／","…"]
    if (inn[0]=='\u3000') or (inn[0]=='\n'):
        bies = ""
        state=0
    elif (inn[0] in punc_list) or (inn[1]=='\u3000'):
      bies="S"
      state=0
    else:
      bies = "B"
      state=1
    
    for i in range(1,len(inn)-1):
        if (inn[i]=='\u3000') or (inn[i]=='\n') :
            continue
        if state ==1:
            
            if (inn[i+1]=='\u3000') or (inn[i+1]=='\n') or (inn[i+1] in punc_list):
                bies = bies + 'E'
                state=0
            else:
                bies = bies + 'I'
        
        elif state ==0:
            if inn[i] in punc_list:
              bies=bies+'S'
            elif (inn[i+1]== '\u3000') or (inn[i+1]=='\n'):
                bies = bies + 'S'
            else:
                bies= bies + 'B'
                state=1
                
    if inn[-1]=='\u3000' or inn[-1]=='\n':
        bies =bies
    else:
        bies =bies +'S'
    return(bies)                
    

def files_creator(original_files, output_text_name='msr_training_', train=True, simplified=[True]):
    """ Creates input file (no space) and Label file (BIES)
    -----------------------------------------------  
    Params:
        original_file (list): The list of original source files for producing input_file and label_file
        output_text_name_name (str): The string to be used as the name of output files 
        train (boolean): If true, unigram and bigram vocabularies will be also created from original file
        simplified (list): List of booleans to specify whether each original_file is simplified Chinese or not
    -----------------------------------------------  
    Returns:
        This function doesn't return anything. It creates files in the resources folder.
    -----------------------------------------------  
    """
    
    line=[]
    for file_no in range(len(original_files)):
        
        with open(original_files[file_no], encoding='utf-8') as file:
            temp_line= file.readlines()
            
        if simplified[file_no]== False:
            temp_line = [hanziconv.HanziConv.toSimplified(l) for l in temp_line]
        line.extend(temp_line)

    line=[spacer(i) for i in line]
    
    input_list = [data_converter(i) for i in line] #apply function can be ued for eff 
    bies_list = [beis_generator(i) for i in line]
    
    
    if  train == True:
        word_occurence_count = {}
        word_to_id_temp= {}
        n=2
        for i in input_list:
            for w in i: 
                if w not in word_to_id_temp.keys():
                    word_to_id_temp[w] = n
                    word_occurence_count[w]=1
                    n+=1
                else:
                    word_occurence_count[w]+=1

        word_to_id={}
        n=2
        for word, occurence in list(word_occurence_count.items()):
           if occurence>=10:
               word_to_id[word] = n
               n+=1

            
                    
        id_to_word = {v:i for i,v in word_to_id.items()}
        
        
        bi_occurence_count = {}
        bi_to_id_temp={}
        idm=4 #0 : padding, 1: <UNK> , 2: <S> , 3:</S>
    
        for li in input_list:
            for i in range(len(li)-1):
                new_bi = li[i]+li[i+1]
                if new_bi not in bi_to_id_temp:
                    bi_to_id_temp[new_bi] = idm
                    bi_occurence_count[new_bi] = 1
                    idm+=1
                else:
                    bi_occurence_count[new_bi]+=1


        bi_to_id={}
        idm=4
        for bi, occurence in list(bi_occurence_count.items()):
           if occurence>=25:
               bi_to_id[bi] = idm
               idm+=1

        id_to_bi = {v: i for i,v in bi_to_id.items() }
        
        
        print("======="*10)
        print('Writing word - id dictionaries!')
        
        word_to_id_path = os.path.join('..', 'resources' , output_text_name +'word_to_id.pkl')
        with open(word_to_id_path, 'wb') as f:
            pickle.dump(word_to_id,f)
            
        bi_to_id_path = os.path.join('..', 'resources', output_text_name+'bi_to_id.pkl')
        with open(bi_to_id_path, 'wb') as f:
            pickle.dump(bi_to_id, f)
        
        print("======="*10)
        print('Unique uni tokens before: ' + str(len(word_to_id_temp.keys())) + 'Unique uni tokens after: ' +str(len(word_to_id.keys())) )
        print('Unique bi tokens before: ' + str(len(bi_to_id_temp.keys())) + 'Unique bi tokens after: ' +str(len(bi_to_id.keys())) )
    
    print("======="*10)
    print('Writing input file and bies file !')
    input_file_path = os.path.join('..', 'resources', output_text_name +'input_file.txt')
    with open(input_file_path, 'w', encoding='utf-8') as f:
        for l in input_list:
            f.write(l+'\n')
    bies_file_path = os.path.join('..', 'resources', output_text_name+'bies_file.txt')
    with open(bies_file_path, 'w' , encoding='utf-8') as f:
        for l in bies_list:
            f.write(l+'\n')
    
    print("======="*10)
    print('Length of files :' + str(len(input_list)))
    print("======="*10)
    