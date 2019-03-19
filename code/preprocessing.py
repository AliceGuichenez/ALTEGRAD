import numpy as np
import networkx as nx
from time import time

import GraphData as data
from walks import generate_walks

from utils import merge_params


def run_preproc(df_name, test = False, pad_vec_idx = 1685894, params = None):
    
    default_params = {
        "max_doc_size" : 70,
        "walk_length" : 10,
        "num_walks" : 5,
        "p" : None,
        "q" : None,
        "N_train" : None,
        "biased" : False,
    }
    params = merge_params(params, default_params)
    
    start_time = time() 
    
    idxs, edgelists = data.get_graphs(N_train = params["N_train"], test = test)
    N = len(idxs)
    
    docs = np.empty((N, params["max_doc_size"], params["walk_length"] + 1), dtype = np.int)
    docs.fill(pad_vec_idx)
    print('document array shape:',docs.shape) # (93719, 70, 11)
    
    for i, edgelist in enumerate(edgelists):
        g = nx.read_edgelist(edgelist) # construct graph from edgelist
        docs[i] = generate_walks(g, params["num_walks"], params["walk_length"],
                             params["max_doc_size"], pad_vec_idx,
                             biased=False, p=params["p"] , q=params["q"]) # create the pseudo-document representation of the graph
        if i % 50 == 0:
            print("Computing graph {}/{}...".format(i, N).ljust(10), end = "\r")
    
    print('documents generated')
    
    data.save_docs(idxs, docs, params, name = df_name)
    
    print('documents saved')
    print('everything done in', round(time() - start_time,2)) # 471.33
    
    return params