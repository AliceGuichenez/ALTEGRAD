# ALTEGRAD
Data challenge: predict continuous values associated with graphs

## Motivation
Find the best HAN architecture in order to predict four continuous targets from graphs

![Graph_pic](./img/graph.png)


## Improvements from the baseline model
- Activation function turned to linear
- Biased random walks (brought no improvements)
- Number of random walks
- Length of random walks
- Enriching nodes attributes through Weisfeiler-Lehman procedure
- Change optimizer for momentum SGD (brought no improvements)
- Hyper-parameters tuning: batch size, dropout rate, learning rate and patience
- Change the embedding with Role2Vec
- Utilities creation in order to improve the ergonomy

## Organisation of the folders
Code folder:
  - Model utility files
  - 2 notebooks presenting descriptive statistics and some hyper-parameters tuning
  - Role2Vec folder: Personalized Role2Vec library files

Data Folder:
  - Empty folders to fill in with the data

## How to use it
Data have not been uploaded in the Repositories (for confidentiality and storage size reasons)
Hence, in order to reproduce our results:
- Clone our [repository](https://github.com/AliceGuichenez/ALTEGRAD)
- Place the data in the data folder
- From the Altegrad/code folder just run:
  ```python
  python3 main.py
  ```
