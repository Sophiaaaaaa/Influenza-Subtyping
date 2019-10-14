# Influenza-Subtyping
  a deep learning method for Influenza Subtyping
  
  This is the source code of the paper named "Rapid Detection and Prediction of Influenza A Subtype Using Deep Convolutional Neural Network Based Ensemble Learning", which proposes a deep convolutional neural network based ensemble learning model to precisely detect all subtypes of influenza A viruses. This code runs with Python v3.5.2 and Keras v2.2.4.

## Influenza data set access
  The influenza sequences in this work are from [NCBI Influenza Virus database](https://www.ncbi.nlm.nih.gov/genomes/FLU/Database/nph-select.cgi#mainform). You should first download HA and NA protein sequences of influenza A.
## How to test the model?
  use the following command to test.
  ```Linux
  python main.py --phase test --type_name ['HA' or 'NA']
  ```
## How to train your own model?
  ```Linux
  python main.py --phase train --type_name ['HA' or 'NA']  --epoch 50 --lr 0.0001
  ```
## Train and Test with your own data 
  You can also train and test with your selected influenza sequences. Delete the file under 'data_set' folder, then copy your .fa file to 'data' folder and set prepareData.path as the path of your .fa file. Then run the above command to train and test your data.
  
## The Guideline of .py file in this repository.
  1. main.py: main funcion.
  2. data_processing.py: preprocess the sequences in the dataset and form the one-hot embedded sequences and labels.
  3. transSeq.py: encoding influenza seqs by one-hot.
  4. data_set.py: form the datasets for ensemble learning.
  5. Resnet.py: Resnet binary sub-classifier of ensamble learning.
  6. train.py: model training process.
  7. test.py: test process.
  8. utils.py: contains some basic functions.
