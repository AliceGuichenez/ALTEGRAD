import numpy as np
import networkx as nx
from time import time
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
docs = []
idxs, edgelists = data.get_graphs(N_train = 5000, test = False)
N = len(idxs)



for i, edgelist in enumerate(edgelists):
    g = nx.read_edgelist(edgelist) # construct graph from edgelist
    doc = generate_walks(g,num_walks,walk_length, biased=True, p =p , q=q) # create the pseudo-document representation of the graph
    docs.append(doc)
    if i % 50 == 0:
        print("Computing graph {}/{}...".format(i, N).ljust(10), end = "\r")

print('documents generated')

# truncation-padding at the document level, i.e., adding or removing entire 'sentences'
docs = [d+[[pad_vec_idx]*(walk_length+1)]*(max_doc_size-len(d)) if len(d)<max_doc_size else d[:max_doc_size] for d in docs] 

docs = np.array(docs).astype('int')
print('document array shape:',docs.shape) # (93719, 70, 11)

data.save_docs(idxs, docs, name = "small_biased")

print('documents saved')
print('everything done in', round(time() - start_time,2)) # 471.33

# = = = = = = = = = = = = = = =






























