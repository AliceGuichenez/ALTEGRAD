
##############################################################

### preprocessing

import os
os.chdir('/Users/aliceguichenez/Documents/Ecoles/Master_X/S1/ALTEGRAD/Challenge/ALTEGRAD/code/')

import re
import numpy as np
import networkx as nx
import random
from HAN import HAN

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
import sys

import json
import GraphData as data

import csv

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

q = 1.2
p = np.random.uniform(min(q,1)-0.3, max(q,1)+0.3)
def compute_probabilities(graph):
    '''Compute a biased matrix of transition for the graph'''
    probs = {}
    for source_node in graph.nodes():
        probs[source_node] = {}
        for current_node in graph.neighbors(source_node):
            probs_ = []
            for destination in graph.neighbors(current_node):
                if source_node==destination:
                    prob_ = graph[current_node][destination].get('weight',1) * (1/p)
                elif destination in graph.neighbors(source_node):
                        prob_ = graph[current_node][destination].get('weight',1)
                else:
                    prob_ = graph[current_node][destination].get('weight',1) * (1/q)
                probs_.append(prob_)
            probs[source_node][current_node] = probs_/np.sum(probs_)
    return probs

def biased_walk(graph, node, walk_length):
    '''Return a biased random walk with probabilities in probs'''
    probs = compute_probabilities(graph)
    walk = [node]
    neighbors = list(graph.neighbors(walk[-1]))
    walk.append(random.choice(neighbors))
    for i in range(walk_length-1):
        neighbors = list(graph.neighbors(walk[-1]))
        probabilities = probs[walk[-2]][walk[-1]]
        walk.append(np.random.choice(neighbors, p=probabilities))
    return walk

def generate_walks(graph, num_walks, walk_length, max_doc_size, padding_filler, biased = False):
    '''samples num_walks walks of length walk_length+1 from each node of graph'''
    # Set the generator of walk (biased or not)
    if biased:
        generator = biased_walk
    else:
        generator = random_walk
    
    # Generate walk for each node
    graph_nodes = graph.nodes()
    walks = np.empty((max_doc_size, walk_length + 1), dtype = np.int)
    walks.fill(padding_filler)
    
    i = 0
    for _ in range(num_walks):
        nodes = np.random.permutation(graph_nodes)
        for node in nodes:
            walk = generator(graph, node, walk_length)
            walks[i] = np.array(walk, dtype = np.int)
            i += 1
            if max_doc_size <= i:
                break
        if max_doc_size <= i:
            break
    return walks

path_root = '/Users/aliceguichenez/Documents/Ecoles/Master_X/S1/ALTEGRAD/Challenge/for_kaggle_final'
path_to_data = path_root + '/data/'

idxs, edgelists = data.get_graphs(N_train = 1000, test = False)
N = len(idxs)


#### RUN FROM HERE

########################################################
### generate documents

for q in [1.2,0.8]:
    for p in [q-0.2, q+0.2]:
        
        mse = []
        mae = []
        pad_vec_idx = 1685894 # 0-based index of the last row of the embedding matrix (for zero-padding)
        
        # parameters
        num_walks = 5
        walk_length = 10
        max_doc_size = 70
        
        docs = np.empty((N, max_doc_size, walk_length + 1), dtype = np.int)
        docs.fill(pad_vec_idx)
        print('document array shape:',docs.shape) # (93719, 70, 11)
        
        for i, edgelist in enumerate(edgelists):
            g = nx.read_edgelist(edgelist) # construct graph from edgelist
            docs[i] = generate_walks(g, num_walks, walk_length,
                                 max_doc_size, pad_vec_idx,
                                 biased=True)
        
            if i % 50 == 0:
                print("Computing graph {}/{}...".format(i, N).ljust(10), end = "\r")
        
        print('documents generated')
        
        data.save_docs(idxs, docs, name = "1000_graphs_biased")
        
        print('documents saved')
        
        ########################################################
        ### test model
        
        
        
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
        
        docs, target = data.get_dataset("1000_graphs_not_WL")
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
        
        ##### read results predict
        
        
        
        
        # = = = = = TRAINING RESULTS = = = = = 
        
        for tgt in range(4):
            
            print('* * * * * * *',tgt,'* * * * * * *')
            
            with open(os.path.join(data.data_path, 'model_history_' + str(tgt) + '.json'), 'r') as file:
                hist = json.load(file)
            
            val_mse = hist['val_loss']
            val_mae = hist['val_mean_absolute_error']
            
            min_val_mse = min(val_mse)
            min_val_mae = min(val_mae)
            
            best_epoch = val_mse.index(min_val_mse) + 1
            
            print('best epoch:',best_epoch)
            print('best val MSE',round(min_val_mse,3))
            print('best val MAE',round(min_val_mae,3))
        
            mse.append(round(min_val_mse,3))
            mae.append(round(min_val_mae,3))
            
        with open(os.path.join(data.data_path, 'mse_p_' + str(p) + '_q_' + str(q) + '.csv'), 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(mse)
        
        with open(os.path.join(data.data_path, 'mae_p_' + str(p) + '_q_' + str(q) + '.csv'), 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(mae)
                
                
                