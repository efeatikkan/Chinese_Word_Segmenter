# Chinese_Word_Segmenter

This is a  Chinese Word Segmenter project that is highly based on "State-of-the-art Chinese Word Segmentation with Bi-LSTMs" paper available
at https://arxiv.org/abs/1808.06511. 
In addition to the base model described in the paper, a Conditional Random Field layer is added on top of the dense layer. 


## Datasets
Academia Sinica, City University of Hong Kong, Peking University and Microsoft Research corpuses (avaliable at 
http://sighan.cs.uchicago.edu/bakeoff2005/ ) are used for creating token vocabularies and training the model. 

## Quick Prediction
predict.py can be used directly with inputs:
  - input_text, chinese txt file without blanks
  - output_text, BIES predictions txt
  - resources_folder, folder that includes the tensorflow trained model and unigram/ bigram token vocabularies

```
predict.py path/input_file.txt path/outputfile.txt path/resourcesfolder
```
