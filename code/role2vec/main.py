from collections import namedtuple
from role_utils import load_graph
import os
import time
from role2vec import Role2Vec
import numpy as np


# Parameters
dim = 20
batch_size = 200 # max 300 for memory
graphs_i = 0
graphs_j = 100000



# Arguments for the embeddings
args_default = {
    "sampling" : "first",
    
    "window_size" : 5,
    "walk_number" : 10,
    "walk_length" : 80,
    "sampling" : "first",
    "P" : 1,
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



# Fetching all graphes
file_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(file_path, "../../data/")
edges_path = os.path.join(data_path, 'edge_lists/')
edgelists = os.listdir(edges_path)
edgelists = [edges_path + edge for edge in edgelists]



embs_file = "embs_{}.npy".format(dim)
embs = np.load(embs_file) if os.path.isfile(embs_file) else np.zeros((1685894, dim))

graphes = edgelists[graphs_i:graphs_j]
N_graphes = len(graphes)
g_batch = None
N_done = 0
t0, T_start = time.process_time(), 0
for i, graph_path in enumerate(graphes):
    # Loading graph and adding it to the batch
    g = load_graph(graph_path)
    if embs[int(list(g.nodes())[0])][0] != 0: # Already computed then skip
        continue
    if g_batch is None:
        g_batch = g
        N_batch = 1
    else:
        g_batch.add_edges_from(g.edges())
        N_batch += 1
    
    # Compute embedding of the batch
    if N_batch == batch_size or i + 1 == N_graphes:
        N_done += N_batch
        print("####### Starting batch {}/{} #######".format((i+1)//batch_size, N_graphes//batch_size))
        print("\tBatch number of nodes : {}".format(len(g_batch)))
        model = Role2Vec(args, graph = g_batch)
        print("\tRandom walks...")
        model.do_walks()
        print("\tStructural features...")
        model.create_structural_features()
        print("\tLearning embedding...")
        model.learn_embedding()
        
        idxs = [int(i) for i in model.graph.nodes]
        embs[idxs, :] = model.embedding
        print("\tSaving...")
        np.save(embs_file, embs) # saving
        
        g_batch = None
        
        T = time.process_time() - t0
        v = round(T/N_batch, 2)
        T_tot = time.process_time() - T_start
        v_mean = T_tot/N_done
        T_estimated = int(round((v_mean*(N_graphes - i - 1))/60))
        T = round(T)
        t0 = time.process_time()
        
        print("####### Done until {} in {} seconds with speed {} s/graph #######".format(i, T, v))
        print("####### Running since {} minutes still {} minutes to go ({} s/graph). #######\n\n".format(int(T_tot/60), T_estimated, round(v_mean, 3)))
              
              

              
    