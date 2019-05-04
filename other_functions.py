# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 14:40:36 2019

@author: Efe
"""
import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import pickle

def batch_gen_bi(x,x2,y, batch_size):
    """ Returns a batch_generator that yields batch sized x and y
    -----------------------------------------------
    Params:
            x  : Unigram representations 
            x2 : Bigram representations
            y  : Ground-truth labels
    batch_size : Batch size for the each yield
    -----------------------------------------------
    Returns:
        A batch generator to be used in the model
    -----------------------------------------------    
    """
    while True:
      all_perm = np.random.permutation(len(x)) 
      for i in range(len(x)//batch_size):
          
          yield( x[all_perm[i*batch_size :(i+1)*batch_size]],x2[all_perm[i*batch_size :(i+1)*batch_size]], y[all_perm[i*batch_size :(i+1)*batch_size]])
 
    
def word_to_id_converter( in_sent, dic):
    """Given a String(line) returns unigram integer id representations by checking it from given
    unigram vocabulary dictionary.
    -----------------------------------------------   
    Params:
        in_sent : String to be transformed into unigram integer id representation
            dic : The vocabulary dictionary to be used as the lookup source for each unigram word - id pair.
                  This dictionary is usually created in the preprocessing phase, from original file.
    -----------------------------------------------              
    Returns:
        List of integer unigram ids corresponding to each unigram chracter in the given string. If a chracter is not
        defined in the given unigram vocabulary dictionary, <UNK> id 1 is given for that chracter.
   ----------------------------------------------- 
    """
    outp= []
    for word in in_sent:
      try:
        outp.append(dic[word])
      except:
        outp.append(1)
    return(outp) 

def bi_to_id_converter(in_sent,dic):
    """Given a String(line) returns bigram integer id representations by checking it from given
    bigram vocabulary dictionary.
    ----------------------------------------------- 
    Params:
        in_sent : String to be transformed into bigram integer id representation
            dic : The vocabulary dictionary to be used as the lookup source for each bigram word - id pair.
                  This dictionary is usually created in the preprocessing phase, from original file.
    -----------------------------------------------              
    Returns:
        List of integer bigram ids corresponding to each bigram chracter in the given string. If a chracter is not
        defined in the given bigram vocabulary dictionary, <UNK> id 1 is given for that chracter. Start of sentence
        token <s> id 2 and end of sentence token <\s> id 3 is added to the begining and end of each string. 
   ----------------------------------------------- 
    """
    outp=[2]
    for i in range(len(in_sent)-1):
      token= in_sent[i]+ in_sent[i+1]
      try:
        outp.append(dic[token])
      except:
        outp.append(1)
    outp.append(3)
    return(outp)
    
def generator_creator(source_name='msr_training_', dict_source_name = 'msr_training_'):
    """ Returns a batch_generator for the given input and label file.
    ----------------------------------------------- 
    Params:
        source_name: Base name for the input and label files. exp: given 'msr_training_' 
                        the function will look for resources/msr_training_input_file.txt and resources/msr_training_bies_file.txt
   dict_source_name: Base name for the unigram and bigram vocabulary dictionaries to transform input file into 
                      list of corresponding id lists. 
    -----------------------------------------------                   
    Returns: 
          data_gen : The batch generator that yields matrices of unigram_representation, bigram_representation
                       and ground_truth BIES labels in batches of 32. 
     uni_dict_size : Vocabulary size of unigram dictionary. This value will be used to in the unigram embedding layer of  the  model.  
      bi_dict_size : Vocabulary size of bigram dictionary. This value will be used to in the bigram embedding layer of  the  model.  
    ----------------------------------------------- 
    Notes: 
        All the unigram and bigram samples are padded to fixed length of 150. If their size were larger, they are truncated.
    -----------------------------------------------     
    """
    
    input_list_path = os.path.join('..', 'resources', source_name + 'input_file.txt')
    with open(input_list_path, 'r', encoding='utf-8') as f:
        input_list = f.readlines()
    
    input_list= [l.replace('\n','') for l in input_list]

    bies_list_path= os.path.join('..', 'resources', source_name + 'bies_file.txt')
    with open(bies_list_path, 'r' , encoding='utf-8') as f:
        bies_list = f.readlines()
    bies_list= [l.replace('\n','') for l in bies_list]
    
    word_to_id_path = os.path.join('..', 'resources', dict_source_name+'word_to_id.pkl')
    with open(word_to_id_path, 'rb') as f:
        word_to_id= pickle.load(f)
     
    bi_to_id_path = os.path.join('..', 'resources', dict_source_name+'bi_to_id.pkl')
    with open(bi_to_id_path, 'rb') as f:
        bi_to_id= pickle.load(f)
    
    
    uni_dict_size = len(word_to_id.items())
    bi_dict_size = len(bi_to_id.items())
    
    
    input_list_vect = [word_to_id_converter(i,word_to_id) for i in input_list]
    input_list_vect2 = [bi_to_id_converter(i,bi_to_id) for i in input_list]
    
    padded_input = pad_sequences(input_list_vect , maxlen=150,  padding='post', truncating='pre')
    padded_input2 = pad_sequences(input_list_vect2 , maxlen=150, padding='post', truncating = 'pre')
    
    
    bies= {'B':0, 'I': 1, 'E': 2, 'S':3}
    
    def bies_converter(in_list):
        "Function to convert BIES encodings to integer representations of 0,1,2,3 "
        outp=[]
        for c in in_list:
            outp.append(bies[c])
        return(outp)
    bies_list_vectorized = [bies_converter(i) for i in bies_list]
    padded_bies_list = pad_sequences(bies_list_vectorized , maxlen=150,  padding='post', truncating='pre')
    
    padded_categorical_bies_list  = to_categorical(padded_bies_list)
    data_gen = batch_gen_bi(padded_input, padded_input2, padded_categorical_bies_list , 32)

    return(data_gen, uni_dict_size, bi_dict_size)

    
def create_tf_model(uni_dict_size, bi_dict_size):
    """Creates tensorflow model and returns necessary operations/ tensors in order to train the model.
    -----------------------------------------------     
    Params:
        uni_dict_size: Unigram vocabulary dictionary size to be used in unigram embedding layer
        bi_dict_size : Bigram vocabulary dictionary size to be used in bigram embedding layer
    -----------------------------------------------         
    Returns:
        list of [train, loss, accuracy ,inputs, inputs_bi, labels, dropout probability ,viterbi output sequence, labels_masked,preds_non_masked]
        These operations/tensors will be input to the training session of tensorflow model.
    -----------------------------------------------     
    Notes:
        General structure of the model created:
            Embeddings > bi-LSTM > Dense > CRF
            
        Embedding sizes (unigram / bigram ) : 128 / 256
        hidden size of LSTM : 256
   -----------------------------------------------          
    """
    vocab_size_uni = uni_dict_size +2
    embedding_size_uni = 128
    hidden_size= 256
    
    vocab_size_bi = bi_dict_size +4
    embedding_size_bi = 256

    inputs = tf.placeholder( tf.int32, shape =[None,None] , name= 'inputs' )
    inputs_bi = tf.placeholder( tf.int32 , shape=[None,None] , name= 'inputs_bi')
    
    labels = tf.placeholder( tf.int64 , shape= [None,None,None] , name='labels')
    keep_prob = tf.placeholder(tf.float32, shape=[] , name= 'keep_prob')
    seq_length = tf.count_nonzero(inputs, axis=-1 , name = 'seq_length')
    
    
    with tf.variable_scope("embeddings_uni"):
        embedding_matrix_uni = tf.get_variable("embeddings", shape=[vocab_size_uni, embedding_size_uni])
        embeddings_uni = tf.nn.embedding_lookup(embedding_matrix_uni, inputs)
      

    with tf.variable_scope("embeddings_bi"):
            embedding_matrix_bi = tf.get_variable("embeddings", shape=[vocab_size_bi, embedding_size_bi])
            embeddings_bi = tf.nn.embedding_lookup(embedding_matrix_bi, inputs_bi)      
          
    with tf.variable_scope("rnn"):
            emb_combined= tf.concat([embeddings_uni, embeddings_bi], axis= -1)
            rnn_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
    
            rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell,
                                                     input_keep_prob=keep_prob,
                                                     output_keep_prob=keep_prob,
                                                     state_keep_prob=keep_prob)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(rnn_cell,rnn_cell, emb_combined, sequence_length=seq_length, dtype=tf.float32)
            outputs = tf.concat(outputs,2)
            
    with tf.variable_scope("dense"):
            logits = tf.layers.dense(outputs, 4, activation=None)
            
    with tf.variable_scope("crf"):
        crf_params = tf.get_variable('crf_params', 
                [4, 4], dtype=tf.float32)
            
    
    with tf.variable_scope("loss_crf"):
        labels2 = tf.argmax(labels, axis=2)
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
            logits, labels2, seq_length,crf_params)

        loss = -tf.reduce_mean(log_likelihood, name= "crf_loss_output")
        
    with tf.variable_scope("preds"):
        
        viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(
                        logits, transition_params, seq_length)
        
        mask= tf.sequence_mask(seq_length)
        preds_non_masked = viterbi_sequence
        preds_masked = tf.boolean_mask(viterbi_sequence,mask)
        preds_identity= tf.identity( preds_masked, name= 'viterbi_labels')
        labels_masked = tf.boolean_mask(labels, mask)
        
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(preds_masked,dtype=tf.int64), tf.argmax(labels_masked,axis=1)), tf.float32), name= 'acc')

    with tf.variable_scope("train"):
        global_step= tf.Variable(0, trainable=False)
        starting_learning_rate = 1e-3
        learning_rate = tf.train.exponential_decay( starting_learning_rate,global_step , 400, 0.90, staircase=True)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)        

    return([train_op, loss, acc,inputs, inputs_bi, labels, keep_prob,viterbi_sequence,labels_masked,preds_non_masked])
    
def run_model(gens_train, gen_dev, train_optimizer_op , loss_op , acc_op , inputs, inputs_bi, labels , keep_prob,preds_masked,labels_masked,preds_non_masked , epochs= 100, iterations= 200, save_model=True):
    """ Trains the model by using input operations/tensors given as input. The model can be saved for the further use.
    -----------------------------------------------   
    Params:
        gens_train : Train batch generator which will be used as the data source for training the model
        gen_dev    : Development batch generator which will be used as the data source for development loss/accuracy calculations
Operations/Tensors : [train_optimizer_op , loss_op , acc_op , inputs, inputs_bi, labels , keep_prob,preds_masked,labels_masked,preds_non_masked] operations to train the model
        Epochs     : Number of epochs for training
        Iterations : Number of iterations per epoch
        Save model : Boolean. If true saves the trained  tensorflow model for further usage.
    -----------------------------------------------       
    Returns:
        This function doesn't return anything. It trains the model, prints training and development losses/ accuracies.
        If save model = True, saves the model.
    -----------------------------------------------       
    Notes: 
        A tensorboard graph output is also created to used for examination. 
    -----------------------------------------------       
    """
    
    epochs = epochs
    batch_size = 32

    n_iterations =iterations
    n_dev_iterations =200
    
    data_train_gen = gens_train
    data_dev_gen  = gen_dev
    
    
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        print("\nStarting training...")
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            print("\nEpoch", epoch + 1)
            epoch_loss, epoch_acc = 0., 0.
            mb = 0
            print("======="*10)
            for _ in range(n_iterations): 
              
                batch_x_uni, batch_x_bi, batch_y= next(data_train_gen)
                mb += 1
                _, loss_val, acc_val = sess.run([train_optimizer_op, loss_op, acc_op], 
                                                feed_dict={inputs: batch_x_uni, inputs_bi:batch_x_bi, labels:batch_y, keep_prob: 0.90})

                epoch_loss += loss_val
                epoch_acc += acc_val
        
            epoch_loss /= n_iterations
            epoch_acc /= n_iterations
            
            summary_train_loss = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=epoch_loss)])
            writer.add_summary(summary_train_loss, global_step=epoch)
            
            summary_train_acc = tf.Summary(value=[tf.Summary.Value(tag='train_acc', simple_value=epoch_acc)])
            writer.add_summary(summary_train_acc, global_step=epoch)
            
            print("\n")
            print("Train Loss: {:.4f}\tTrain Accuracy: {:.4f}".format(epoch_loss, epoch_acc))
            print("======="*10)
            
            dev_loss=0
            dev_acc=0
            for _ in range(n_iterations):
                batch_x_uni_dev, batch_x_bi_dev, batch_y_dev= next(data_dev_gen)
                loss_val, acc_val , preds_h, labels_h= sess.run([loss_op, acc_op , preds_non_masked,labels], 
                                                feed_dict={inputs: batch_x_uni_dev, inputs_bi:batch_x_bi_dev, labels:batch_y_dev, keep_prob: 1.})
            
                dev_loss += loss_val
                dev_acc += acc_val
                

                
            final_loss= dev_loss/ n_iterations
            final_acc= dev_acc/ n_iterations
             
            
            summary_dev_loss = tf.Summary(value=[tf.Summary.Value(tag='dev_loss', simple_value=final_loss)])
            writer.add_summary(summary_dev_loss, global_step=epoch)
            
            summary_dev_acc = tf.Summary(value=[tf.Summary.Value(tag='dev_acc', simple_value=final_acc)])
            writer.add_summary(summary_dev_acc, global_step=epoch)
            
            print("\n")
            print("Development Loss: {:.4f}\tDevelopment Accuracy: {:.4f}".format(final_loss, final_acc))
            print("======="*10)

        if save_model==True:
            print("Saving Model!")
            print("======="*10)
            saver= tf.train.Saver()
            model_save_path = os.path.join('..','resources', 'the_model')
            saver.save(sess, model_save_path)
            

        
def predict(test_gen):
    with tf.Session() as sess:
        saved_model= tf.train.import_meta_graph('the_model.meta')
        saved_model.restore(sess, tf.train.latest_checkpoint(''))
        graph = tf.get_default_graph()
        inputs = graph.get_tensor_by_name('inputs:0')
        inputs_bi = graph.get_tensor_by_name('inputs_bi:0')
        labels = graph.get_tensor_by_name('labels:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        
        loss_outp = graph.get_tensor_by_name('loss/loss_output:0')
        preds_non_masked = graph.get_tensor_by_name('loss/preds_non_masked:0')
        acc= graph.get_tensor_by_name('loss/acc:0')
        
        total_loss, total_acc= 0. , 0.
        c= 0 
        for _ in range(10):
            batch_x_uni, batch_x_bi, batch_y= next(test_gen)
            loss_val, acc_val , preds_h, labels_h= sess.run([loss_outp, acc , preds_non_masked,labels], 
                                                feed_dict={inputs: batch_x_uni, inputs_bi:batch_x_bi, labels:batch_y, keep_prob: 1.})
            c+=1
            total_loss += loss_val
            total_acc += acc_val
        
        final_loss= total_loss/ c
        final_acc= total_acc/ c
        
        print('loss_test: ' + str(final_loss))
        print('acc_test: ' + str(final_acc))
        print("======="*10)
    
def reverse_bies(inn, seq_len):
    """Given a list of BEIS corresponding id representations, returns a string of BEIS encoding
    string. The output is truncated by the given seq_len parameter. """
    r_bies = {0:'B',  1:'I',  2:'E', 3:'S'}
    inn= inn[:seq_len]
    outp = [r_bies[i] for i in inn]
    outp= ''.join(outp)
    return(outp)
