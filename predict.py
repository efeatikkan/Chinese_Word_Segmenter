from argparse import ArgumentParser
import other_functions
import os 
import tensorflow as tf 
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", help="The path of the input file")
    parser.add_argument("output_path", help="The path of the output file")
    parser.add_argument("resources_path", help="The path of the resources needed to load your model")

    return parser.parse_args()


def lines_sepearat_map(inn):
    dd ={}
    for i in range(len(inn)):
        if len(inn[i])>150:
            no_of_new_lines = np.ceil(len(inn[i])/150)
            dd[i]=no_of_new_lines
    return(dd)

def predict(input_path, output_path, resources_path):
    """
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the BIES format.
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.

    :param input_path: the path of the input file to predict.
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """


    """In this script the given text will be transformed to a list, after 
    preprocessing, their BIES encodings will be predicted line by line.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        input_list = f.readlines()
    input_list= [l.replace('\n','') for l in input_list]
        
    word_to_id_path = os.path.join(resources_path, 'all_training_word_to_id.pkl')
    with open(word_to_id_path, 'rb') as f:
        word_to_id= pickle.load(f)

    bi_to_id_path = os.path.join( resources_path, 'all_training_bi_to_id.pkl')
    with open(bi_to_id_path, 'rb') as f:
        bi_to_id= pickle.load(f)
        
        
        
    """Lines longer than 150 appended to the end of the input list and will be 
    predicted seperately. In the end they will concetanated back their original
    positions.
    """
    lines_seperated_dict = lines_sepearat_map(input_list)
    sep_tuple = [(a,b) for a,b in lines_seperated_dict.items()]
    sep_tuple.sort()
    
    for key,value in sep_tuple:
        initial_sentence = input_list[key]
        for i in range(int(value)):
            start_index = i*150
            end_index = (i+1)*150
            if i==int(value)-1:
                end_index = start_index+len(initial_sentence)%150
            input_list.append( initial_sentence[start_index : end_index])

        
    for i,(key,value) in enumerate(sep_tuple):
        input_list.pop(key-i)

    input_list_vect = [other_functions.word_to_id_converter(i,word_to_id) for i in input_list]
    input_list_vect2 = [other_functions.bi_to_id_converter(i,bi_to_id) for i in input_list] 
    
    
    padded_input = pad_sequences(input_list_vect , maxlen=150,  padding='post', truncating='pre')
    padded_input2 = pad_sequences(input_list_vect2 , maxlen=150, padding='post', truncating = 'pre')
    
    with tf.Session() as sess:
        saved_model_path = os.path.join(resources_path, 'the_model.meta')
        
        saved_model= tf.train.import_meta_graph(saved_model_path)
        saved_model.restore(sess, tf.train.latest_checkpoint(resources_path))
        graph = tf.get_default_graph()
        
        inputs = graph.get_tensor_by_name('inputs:0')
        inputs_bi = graph.get_tensor_by_name('inputs_bi:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        preds_non_masked = graph.get_tensor_by_name('preds/viterbi_labels:0')

        predictions=[]

        new_sentences_len = np.sum(list(lines_seperated_dict.values()))
        new_sentences_start_index  =int(len(input_list) -new_sentences_len )
        for i in range(len(padded_input2)):
            if i%1000==0:
                print(i)
            batch_x_uni, batch_x_bi= padded_input[i].reshape((1,150)), padded_input2[i].reshape((1,150))
            
            prediction = sess.run(preds_non_masked, feed_dict={inputs:batch_x_uni , inputs_bi: batch_x_bi , keep_prob:1})
            prediction_reversed = other_functions.reverse_bies(prediction.reshape(-1),len(input_list_vect[i]))
            predictions.append(prediction_reversed)
        
        pred2 = predictions.copy()
        for i,(key,value)  in enumerate(sep_tuple):
            concat_prediction = ''
            for _ in range(int(value)):

                concat_prediction = concat_prediction +  pred2.pop(new_sentences_start_index+i)
            pred2.insert(key, concat_prediction)
            
        with open(output_path, 'w') as f:
            for l in pred2:
                f.write(l+'\n')
            
            



if __name__ == '__main__':
    args = parse_args()
    predict(args.input_path, args.output_path, args.resources_path)
