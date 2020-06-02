# Influenza-Subtyping
  a deep learning method for Influenza Subtyping
  
  This is the source code of the paper named "Rapid Detection and Prediction of Influenza A Subtype Using Deep Convolutional Neural Network Based Ensemble Learning", which proposes a deep convolutional neural network based ensemble learning model to precisely detect all subtypes of influenza A viruses. This code runs with Python v3.5.2 and Keras v2.2.4.

This work has been published on [ICBBB '20: Proceedings of the 2020 10th International Conference on Bioscience, Biochemistry and Bioinformatics](https://dl.acm.org/doi/proceedings/10.1145/3386052), if you use the code, please cite the following paper:
```
@inproceedings{10.1145/3386052.3386053,
author = {Wang, Yu and Bao, Junpeng and Du, Jianqiang and Li, Yongfeng},
title = {Rapid Detection and Prediction of Influenza A Subtype Using Deep Convolutional Neural Network Based Ensemble Learning},
year = {2020},
isbn = {9781450376761},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3386052.3386053}, 
doi = {10.1145/3386052.3386053},
booktitle = {Proceedings of the 2020 10th International Conference on Bioscience, Biochemistry and Bioinformatics},
pages = {47–51},
numpages = {5},
keywords = {Virus subtyping, Influenza A viruses, Ensemble learning, Convolutional neural network},
location = {Kyoto, Japan},
series = {ICBBB ’20}
}
```
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
