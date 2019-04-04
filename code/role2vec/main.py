from collections import namedtuple
from role_utils import load_graph
import os
import time
from role2vec import Role2Vec
import numpy as np
import re

# 'atoi' and 'natural_keys' taken from: https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside     
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]    



#############################
#
# CHANGE THE PARTITION HERE :)
#
##############################
graphs_i, graphs_j = 0, 100000 # Nicolas
#graphs_i, graphs_j = 35001, 70000 # Robin
#graphs_i, graphs_j = 70001, 100000 # Alice





# Parameters
dim = 30 # Dimension of the embedding
batch_size = 300 # The number of graph saved together


print("####### WORKING ON PARTITION FROM {} TO {} #######".format(graphs_i, graphs_j))


# Arguments for the embeddings
args_default = {
    "sampling" : "second",
    
    "window_size" : 5,
    "walk_number" : 5,
    "walk_length" : 10,
    "P" : 0.67,
    "Q" : 1,
    
    "dimensions" : dim,
    "down_sampling" : 0.001,
    "alpha" : 0.025,
    "min_alpha" : 0.025,
    "min_count" : 1,
    "workers" : 4,
    "epochs" : 10,
    
    "features" : "wl",
    "labeling_iterations" : 2,
    "log_base" : 1.5,
    "graphlet_size" : 4,
    "uantiles" : 5,
    "motif_compression" : "string",
    #"seed"  : 42, ## WARNING TO CHANGE !!!!
    "factors" : 8,
    "clusters" : 50,
    "beta" : 0.01,
}
args = namedtuple("args", args_default.keys())(*args_default.values())



#Fetching all graphes
file_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(file_path, "../../data/")
edges_path = os.path.join(data_path, 'edge_lists/')
edgelists = os.listdir(edges_path)
edgelists.sort(key=natural_keys) # important to maintain alignment with the targets!
edgelists = [edges_path + edge for edge in edgelists]



embs_file = "embs_{}_v3.npy".format(dim)
print(os.path.isfile(embs_file))
embs = np.load(embs_file) if os.path.isfile(embs_file) else np.zeros((1685895, dim))

graphes = edgelists[graphs_i:graphs_j+1]
N_graphes = len(graphes)
g_batch = None
N_done, N_batch, skipped = 0, 0, 0
t_start = time.process_time()
graphes = graphes[:1]
for i, graph_path in enumerate(graphes):
    # Loading graph and adding it to the batch
    g = load_graph(graph_path)
    if (np.abs(embs[int(list(g.nodes())[0])]).sum() > 0):
        continue
    
    
    # Compute embedding of the graph
    model = Role2Vec(args, graph = g)
    model.do_walks()
    model.create_structural_features()
    model.learn_embedding()
    idxs = [int(i) for i in model.graph.nodes()]
    embs[idxs, :] = model.embedding
    N_batch += 1
    N_done += 1
    
    if (N_batch == batch_size or i + 1 == N_graphes) and N_batch > 0:    
        np.save(embs_file, embs) # saving 
        T_tot = time.process_time() - t_start
        v_mean = T_tot/N_done
        T_estimated = int(round((v_mean*(N_graphes - i - 1))/60))
        print("####### Done until {} with speed {} s/graph #######".format(graphs_i + i, v_mean))
        print("####### Running since {} minutes still {} minutes to go. #######\n\n".format(int(T_tot/60), T_estimated))
        N_batch = 0
        
    # Reset parameters
    t0 = time.process_time()
    
    
          
T_tot = time.process_time() - t_start                      
print("######## YOU HAVE FINISHED WELL DONE AFTER {} minutes !! #######".format(int(T_tot/60)))
print("####### PARTITION FROM {} TO {} IS DONE !! #######".format(graphs_i, graphs_j))
    