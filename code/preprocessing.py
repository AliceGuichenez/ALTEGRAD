import numpy as np
import networkx as nx
from time import time

def padding(d, pad_vec_idx, max_doc_size, walk_length):
    return d+[[pad_vec_idx]*(walk_length+1)]*(max_doc_size-len(d)) if len(d)<max_doc_size else d[:max_doc_size]



# = = = = = = = = = = = = = = =
import GraphData as data
from walks import generate_walks


def run_preproc(df_name, biased, p = None, q = None, N_train = None, test = False,
                max_doc_size = 70, walk_length = 10, num_walks = 5, pad_vec_idx = 1685894):
    
    start_time = time() 
    
    idxs, edgelists = data.get_graphs(N_train = N_train, test = test)
    N = len(idxs)
    
    docs = np.empty((N, max_doc_size, walk_length + 1), dtype = np.int)
    docs.fill(pad_vec_idx)
    print('document array shape:',docs.shape) # (93719, 70, 11)
    
    for i, edgelist in enumerate(edgelists):
        g = nx.read_edgelist(edgelist) # construct graph from edgelist
        docs[i] = generate_walks(g, num_walks, walk_length,
                             max_doc_size, pad_vec_idx,
                             biased=False, p =p , q=q) # create the pseudo-document representation of the graph
        if i % 50 == 0:
            print("Computing graph {}/{}...".format(i, N).ljust(10), end = "\r")
    
    print('documents generated')
    
    data.save_docs(idxs, docs, name = df_name)
    
    print('documents saved')
    print('everything done in', round(time() - start_time,2)) # 471.33





























