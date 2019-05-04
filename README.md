# Chinese Word Segmenter

This is a  Chinese Word Segmenter project that is highly based on "State-of-the-art Chinese Word Segmentation with Bi-LSTMs" paper available
at https://arxiv.org/abs/1808.06511. 
In addition to the base model described in the paper, a Conditional Random Field layer is added on top of the dense layer. 


## Datasets
Academia Sinica, City University of Hong Kong, Peking University and Microsoft Research corpuses (avaliable at 
http://sighan.cs.uchicago.edu/bakeoff2005/ ) are used for creating token vocabularies and training the model. 

## Pretrained Model
The pretrained model and token vocabularies to make predictions can be accesed from this link https://drive.google.com/open?id=1rx4zTPL1hGMbk3Zw2BdOV7IjKXGsQqgG .

## Quick Prediction
After downloading pretrained model, or training the model from scratch, new predictions can be made using
predict.py with the inputs:
  - input_text, chinese txt file without blanks
  - output_text, BIES predictions txt
  - resources_folder, folder that includes the tensorflow trained model and unigram/ bigram token vocabularies

```
predict.py path/input_file.txt path/outputfile.txt path/resourcesfolder
```

