# = = = = = = = = = = = = = = = = = = = = = = = = = = = =
#
# The module GraphData has been designed to make easier the handling of graph data.
# It assures that a graph will be always linked to its right target.
# Basically it always deals with a list of idx and a list of graph together.
# Thus you can always know which graph has which target
#
# = = = = = = = = = = = = = = = = = = = = = = = = = = = =



import os
import re
import pandas as pd
import numpy as np
import pickle
from utils import random_id, merge_params
import datetime
  
# 'atoi' and 'natural_keys' taken from: https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside     
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]        




# = = = = = = = = = = = = = = = = = = = = = = = = = = = =
#
# Global variables for the module (mainly the path to the data)
#
# = = = = = = = = = = = = = = = = = = = = = = = = = = = =


# Get the path of the data folder relative to this file      
# No need to provide it for the user
file_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(file_path, "../data/")

# Store the edgelist
edges_path = os.path.join(data_path, 'edge_lists/')
edgelists = os.listdir(edges_path)
edgelists.sort(key=natural_keys) # important to maintain alignment with the targets!
N_edges = len(edgelists)

target_train = None # A container to load the targets only once





# = = = = = = = = = = = = = = = = = = = = = = = = = = = =
#
# Idxs handling
#
# = = = = = = = = = = = = = = = = = = = = = = = = = = = =  
    
def _read_txt_idxs(filename):
    '''Return an ndarray of int from a text file'''
    with open(os.path.join(data_path, filename), 'r') as file:
        idxs = file.read().splitlines()
    return [int(idx) for idx in idxs]

def get_test_idxs():
    '''Return a list of the idxs of the test dataset (Kaggle)'''
    return _read_txt_idxs("test_idxs.txt")

def get_train_idxs():
    '''Return a list of the idxs of the train dataset'''
    return _read_txt_idxs("train_idxs.txt")


def remove_idxs(idxs, docs, idxs_selected):
    '''Remove the idxs from idxs_selected. It removes the appropriate documents in docs
       Return idxs, docs filtered'''
    idxs = np.array(idxs, dtype = np.int)
    idxs_selected = np.array(idxs_selected, dtype = np.int)
    selector = ~np.isin(idxs, idxs_selected)
    return list(idxs[selector]), docs[selector]
    

# = = = = = = = = = = = = = = = = = = = = = = = = = = = =
#
# Data loading function (dataset, target, embedding)
#
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = 

def get_dataset(name):
    '''Load a specific dataset of document with their features and their target
        It removes the document that are from the test set as we do not have their target'''
    idxs, docs, params = get_docs(name) # Load the raw docs
    idxs, docs = remove_idxs(idxs, docs, get_test_idxs()) # Remove the test docs as we do not have the target for them
    target = get_target(idxs) #Get the target for the docs
    return docs, target, params

def get_kaggle_docs(name):
    idxs, docs, params = get_docs(name) # Load the raw docs
    idxs, docs = remove_idxs(idxs, docs, get_train_idxs()) # Remove the test docs as we do not have the target for them
    return docs[np.argsort(idxs), :, :] # Sorting to get the Kaggle order
    

def get_target(idxs = None):
    '''Return the multiple target as a dataframe with the index of the train graph. It is stored in cache.
        You can ask for specific graph ids with idxs by default it will return the full train dataset'''
    global target_train
    if target_train is None:
        df = pd.DataFrame(index = get_train_idxs())
        for col in range(4):
            with open(os.path.join(data_path, 'targets/train/target_' + str(col) + '.txt'), 'r') as file:
                target = file.read().splitlines()
                df[col] = np.array(target, dtype = np.float)
        target_train = df
    return target_train if idxs is None else target_train.loc[idxs]
    

def get_embeddings():
    '''Return the embeddings data'''
    return np.load(data_path + 'embeddings.npy')

# = = = = = = = = = = = = = = = = = = = = = = = = = = = =
#
# Function to handle the edgelists
#
# = = = = = = = = = = = = = = = = = = = = = = = = = = = =  
    
def get_idxs_graph(idxs):
    '''Return the path for each edgelist with an idx in idxs'''
    return [edges_path + edgelists[idx] for idx in idxs]
    

def get_graphs(N_train = None, test = False):
    '''Generate a list of graph with N_train graph picked randomly from the train dataset
       If test is set to true it adds the test graphs.
       It is usefull if you want to only work on a small portion of the train dataset
       Return idxs, graph_path'''       
    # Pick random edges from train
    idxs = list(np.random.choice(get_train_idxs(), size = N_train, replace = False)) if not(N_train is None) else get_train_idxs()
    
    if test:
        idxs += get_test_idxs() 
    
    return idxs, get_idxs_graph(idxs)


# = = = = = = = = = = = = = = = = = = = = = = = = = = = =
#
# Docs saving and loading (idxs, docs)
#
# = = = = = = = = = = = = = = = = = = = = = = = = = = = =   
    
def get_docs_path(name):
    '''Return the path with an appriate name for the docs to be stored'''
    return os.path.join(data_path, "datasets/docs_{}.pickle".format(name))


from utils import filter_dict
import preprocessing as prec

def save_docs(idxs, docs, params, name):
    '''Save the idxs, docs with pickle.
       Then you can load it again by using get_docs with name'''
    docs = np.array(docs).astype(np.int)
    idxs = np.array(idxs).astype(np.int)
    params = filter_dict(params, prec.default_params.keys()) # only save parameters relevant to the preproc
    params["docs_id"] = random_id()
    file_path = get_docs_path(name)
    file = open(file_path, "wb" )
    pickle.dump((idxs, docs, params),  file)
    file.close()

        

def get_docs(name):
    '''Load (idxs, docs) from the pickle named name'''
    file_path = get_docs_path(name)
    if os.path.isfile(file_path):
        file = open(file_path, "rb" )
        idxs, docs, params = pickle.load(file)
        file.close()
        return idxs, docs, params
    else:
        raise ValueError("The dataset named '{}' cannot be found ! You must generate it first.\nIt could not be find in '{}'.".format(name, file_path))
        
        
def save_perf(params, scores, tgt):
    # Building the performance dictionary
    perf = params.copy()
    perf["target"] = tgt
    now = datetime.datetime.now()
    perf["date"] = now.strftime("%Y-%m-%d %H:%M")
    perf = merge_params(perf, scores)
    new_perf = pd.DataFrame.from_dict(perf, orient = "index").T
    
    # Saving to the other performance
    perfs = get_perfs()
    perfs = perfs.append(new_perf, ignore_index = True)
    perfs.to_csv(get_perf_path(), index = False)
    
    return new_perf
    
def get_perf_path():
    return os.path.join(data_path, "performances.csv")

def get_perfs(train_id = None):
    path_file = get_perf_path()
    perfs = pd.read_csv(path_file) if os.path.isfile(path_file) else pd.DataFrame()
    return perfs if train_id is None else perfs.loc[perfs.train_id == train_id]
    
    
    
    
    
    
    
    
    
    