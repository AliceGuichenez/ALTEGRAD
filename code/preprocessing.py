import numpy as np
import networkx as nx
from time import time

def padding(d, pad_vec_idx, max_doc_size, walk_length):
    return d+[[pad_vec_idx]*(walk_length+1)]*(max_doc_size-len(d)) if len(d)<max_doc_size else d[:max_doc_size]
   
# = = = = = = = = = = = = = = =

pad_vec_idx = 1685894 # 0-based index of the last row of the embedding matrix (for zero-padding)

# parameters
num_walks = 5
walk_length = 10
max_doc_size = 70 # maximum number of 'sentences' (walks) in each pseudo-document
biased = True

# = = = = = = = = = = = = = = =
import GraphData as data
from walks import generate_walks


start_time = time() 
q = 1.2
p = np.random.uniform(min(q,1)-0.3, max(q,1)+0.3)
idxs, edgelists = data.get_graphs(N_train = None, test = True)
N = len(idxs)


docs = np.empty((N, max_doc_size, walk_length + 1), dtype = np.int)
docs.fill(pad_vec_idx)
print('document array shape:',docs.shape) # (93719, 70, 11)

for i, edgelist in enumerate(edgelists):
    g = nx.read_edgelist(edgelist) # construct graph from edgelist
    docs[i] = generate_walks(g, num_walks, walk_length,
                         max_doc_size, pad_vec_idx,
                         biased=True, p =p , q=q) # create the pseudo-document representation of the graph
    if i % 50 == 0:
        print("Computing graph {}/{}...".format(i, N).ljust(10), end = "\r")

print('documents generated')

data.save_docs(idxs, docs, name = "full_biased")

print('documents saved')
print('everything done in', round(time() - start_time,2)) # 471.33

# = = = = = = = = = = = = = = =






























