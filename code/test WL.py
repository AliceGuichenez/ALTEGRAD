
##############################################################

### preprocessing

import os
os.chdir('/Users/aliceguichenez/Documents/Ecoles/Master_X/S1/ALTEGRAD/Challenge/ALTEGRAD/code/')

import re
import numpy as np
import networkx as nx
from collections import defaultdict
import copy
import random
from HAN import HAN

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

def random_walk(graph,node,walk_length):
    walk = [node]
    for i in range(walk_length):
        neighbors = graph.neighbors(walk[i])
        walk.append(random.choice(list(neighbors)))
    return walk

def generate_walks(graph,num_walks,walk_length,biased=False):
    '''
    samples num_walks walks of length walk_length+1 from each node of graph
    '''
    if not biased:
        graph_nodes = graph.nodes()
        n_nodes = len(graph_nodes)
        walks = []
        for i in range(num_walks):
            nodes = np.random.permutation(graph_nodes)
            for j in range(n_nodes):
                walk = random_walk(graph, nodes[j], walk_length)
                walks.append(walk)
        return walks

path_root = '/Users/aliceguichenez/Documents/Ecoles/Master_X/S1/ALTEGRAD/Challenge/for_kaggle_final'
path_to_data = path_root + '/data/'

import GraphData as data
idxs, edgelists = data.get_graphs(N_train = 1000, test = False)
N = len(idxs)

edgelists.sort(key=natural_keys) 

graphs = [nx.read_edgelist(edgelists[i]) for i in range(len(edgelists))]

#################################################
### relabelling

ind = 0
labels = {}
label_lookup = {}
label_counter = 0
N = len(graphs)
h = 5 # ??? nb of iterations, to change
orig_graph_map = {it: {i: defaultdict(lambda: 0) for i in range(N)} for it in range(-1, h)}

for G in graphs:
	for node in G.nodes():
		G.node[node]['label'] = G.degree(node)

# initial labeling
for G in graphs:
    labels[ind] = np.zeros(G.number_of_nodes(), dtype = np.int32)
    node2index = {}
    for node in G.nodes():
        node2index[node] = len(node2index)

    for node in G.nodes():
        label = G.node[node]['label']
        if label not in label_lookup:
            label_lookup[label] = len(label_lookup)

        labels[ind][node2index[node]] = label_lookup[label]
        orig_graph_map[-1][ind][label] = orig_graph_map[-1][ind].get(label, 0) + 1

    ind += 1

compressed_labels = copy.deepcopy(labels)

# WL iterations
for it in range(h):
    unique_labels_per_h = set()
    label_lookup = {}
    ind = 0
    for G in graphs:
        node2index = {}
        for node in G.nodes():
            node2index[node] = len(node2index)

        for node in G.nodes():
            node_label = tuple([labels[ind][node2index[node]]])
            neighbors = G.neighbors(node)
            if len(neighbors) > 0:
                neighbors_label = tuple([labels[ind][node2index[neigh]] for neigh in neighbors])
                node_label =  str(node_label) + "-" + str(sorted(neighbors_label))
            if node_label not in label_lookup:
                label_lookup[node_label] = len(label_lookup)

            compressed_labels[ind][node2index[node]] = label_lookup[node_label]
            orig_graph_map[it][ind][node_label] = orig_graph_map[it][ind].get(node_label, 0) + 1

        ind +=1

    labels = copy.deepcopy(compressed_labels)
    

ix = 0
for G in graphs:
    ixx = 0
    for node in G.nodes():
        G.node[node]['label'] = labels[ix][ixx]
        ixx += 1
    ix += 1


########################################################
### generate documents

pad_vec_idx = 1685894 # 0-based index of the last row of the embedding matrix (for zero-padding)

# parameters
num_walks = 5
walk_length = 10
max_doc_size = 70

docs = []
for idx in range(len(edgelists)):
    g = graphs[idx] # construct graph from edgelist
    doc = generate_walks(g,num_walks,walk_length) # create the pseudo-document representation of the graph
    docs.append(doc)
    
    if idx % round(len(edgelists)/10) == 0:
        print(idx)

print('documents generated')

data.save_docs(idxs, docs, name = "1000_graphs_WL")

print('documents saved')

########################################################
### test model

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import optimizers
import json
import sys

# = = = = = = = = = = = = = = =

is_GPU = False
save_weights = True
save_history = True

path_root = '/Users/aliceguichenez/Documents/Ecoles/Master_X/S1/ALTEGRAD/Challenge/for_kaggle_final'
path_to_code = path_root + '/code/'
path_to_data = path_root + '/data/'

sys.path.insert(0, path_to_code)


# = = = = = hyper-parameters = = = = =
from sklearn.model_selection import train_test_split
    
n_units = 50
drop_rate = 0.3 # modif 1
batch_size = 96
nb_epochs = 10
#modif2:
learning_rate = 0.01
decay_rate = learning_rate / nb_epochs
momentum = 0.8
sgd = optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=True)
my_optimizer = sgd # modif 2

my_patience = 4

docs, target = data.get_dataset("1000_graphs")
X_train, X_test, y_train, y_test = train_test_split(docs, target, test_size=0.3)
embeddings = data.get_embeddings()

import time

res = []
T = 0 # Total time of execution (without cooldown)
t0 = time.process_time()
for tgt in range(4):
    model = HAN(embeddings, docs.shape, is_GPU = False, activation = "linear")
    
    learning_rate = 0.01
    nb_epochs = 10
    momentum = 0.8
    optimizer = 'adam'
    
    if optimizer=='sgd':
        decay_rate = learning_rate / nb_epochs
        my_optimizer = optimizers.SGD(lr=learning_rate, decay=decay_rate, momentum=0.9, nesterov=True)
    elif optimizer=='adam':
        my_optimizer = optimizers.Adam(lr=learning_rate, decay=0)
    
    model.compile(loss='mean_squared_error',
                  optimizer=my_optimizer, metrics=['mae'])
    
    
    
    # = = = = = training = = = = =
    
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=my_patience,
                                   mode='min')
    
    # save model corresponding to best epoch
    checkpointer = ModelCheckpoint(filepath=os.path.join(data.data_path, 'model_' + str(tgt)), 
                                   verbose=1, 
                                   save_best_only=True,
                                   save_weights_only=True)
    
    if save_weights:
        my_callbacks = [early_stopping,checkpointer]
    else:
        my_callbacks = [early_stopping]
    
    model.fit(X_train, 
              y_train[tgt],
              batch_size = batch_size,
              epochs = nb_epochs,
              validation_data = (X_test,y_test[tgt]),
              callbacks = my_callbacks)
    
    hist = model.history.history
    res.append(min(hist["val_loss"]))

    if save_history:
        with open(os.path.join(data.data_path, 'model_history_' + str(tgt) + '.json'), 'w') as file:
            json.dump(hist, file, sort_keys=False, indent=4)
            
    T = time.process_time() - t0
    print("################ {} minutes spent...###########".format(round(T/60)))
