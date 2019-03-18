import sys
import json
import time
import datetime
import pandas as pd
import numpy as np
import os

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers



# = = = = = data loading = = = = =

from sklearn.model_selection import train_test_split
import GraphData as data
from HAN import HAN

def run_training(df_name, model_name,
                 batch_size = 80, my_patience = 2, is_GPU = True,
                 activation = "linear", nb_epochs = 10, 
                 optimizer = "adam", learning_rate = 0.01):
    
    docs, target = data.get_dataset(df_name)
    X_train, X_test, y_train, y_test = train_test_split(docs, target, test_size=0.3)
    
    # = = = = = fitting the model on 4 targets = = = = =
    
    
    res = []
    T = 0 # Total time of execution (without cooldown)
    t0 = time.process_time()
    
    # Building the models
    embeddings = data.get_embeddings()
    model = HAN(embeddings, docs.shape, is_GPU = is_GPU, activation = activation)
    
    if optimizer=='sgd':
        decay_rate = learning_rate / nb_epochs
        my_optimizer = optimizers.SGD(lr=learning_rate, decay=decay_rate, momentum=0.9, nesterov=True)
    elif optimizer=='adam':
        my_optimizer = optimizers.Adam(lr=learning_rate, decay=0)    
        
    model.compile(loss='mean_squared_error',
                  optimizer=my_optimizer, metrics=['mae'])

    # Training for each target
    for tgt in range(4):
        
        
        # = = = = = training = = = = =
        
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=my_patience,
                                       mode='min')
        
        # save model corresponding to best epoch
        model_file = os.path.join(data.data_path, "models/", "{}_{}_{}_model.h5".format(model_name, df_name, tgt))
        checkpointer = ModelCheckpoint(filepath=model_file, 
                                       verbose=1, 
                                       save_best_only=True,
                                       save_weights_only=True)
        
        my_callbacks = [early_stopping,checkpointer]
        
        model.fit(X_train, 
                  y_train[tgt],
                  batch_size = batch_size,
                  epochs = nb_epochs,
                  validation_data = (X_test,y_test[tgt]),
                  callbacks = my_callbacks)
        
        hist = model.history.history
        print(len(hist["val_loss"]))
        res.append(min(hist["val_loss"]))
        
        history_file = os.path.join(data.data_path, "models/", "{}_{}_{}_history.json".format(model_name, df_name, tgt))
        with open(history_file, 'w') as file:
            json.dump(hist, file, sort_keys=False, indent=4)
                
        T = time.process_time() - t0
        print("################ {} minutes spent...###########".format(round(T/60)))
              
    print("##### Summary of the minimal val_loss : {} #####".format(res))              
    return res 




































#l_rate, drop, batch_size, n_units = 0.777003, 0.327482, 139, 82
#run(l_rate, drop, batch_size, n_units)
#
#
#
## = = = = = Randown Search = = = = =
#
#res = [] # Each run results container
#T = 0 # Total time of execution (without cooldown)
#for i_run in []:
#    # Picking random parameters
#    l_rate = np.random.uniform(0.01, 0.8)
#    drop = np.random.uniform(0.2, 0.4)
#    batch_size = int(np.random.uniform(40, 150))
#    n_units = int(np.random.uniform(30, 90))
#
#    # Running the model and storing it
#    t0 = time.time()
#    print("########## RUN {} for : {} #############".format(i_run, (l_rate, drop, batch_size, n_units)))
#    res = run(l_rate, drop, batch_size, n_units)
#    t = time.time() - t0
#    res = [l_rate, drop, batch_size, n_units, t] + res
#    res.append(res)
#
#    # Saving the results at each step
#    df = pd.DataFrame(res, columns = ["l_rate", "drop", "batch_size","n_units", "T", "target0", "target1", "target2", "target3"])
#    df.to_csv("grid_search.csv")
#    T += t
#    print("################ {} minutes spent...###########".format(round(T/60)))
#    time.sleep(300) # 5 minutes cooldown for the GPU
