import sys
import json
import time
import datetime
import pandas as pd
import numpy as np
import os

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers

from utils import merge_params
from scores_kaggle import get_scores

# = = = = = data loading = = = = =

from sklearn.model_selection import train_test_split
import GraphData as data
from HAN import HAN
from utils import random_id

def run_training(df_name, model_name, is_GPU = True, params = None):
    
    default_params = {
        "nb_epochs" : 10,
        "my_patience" : 4,
        "batch_size" : 80,
        "optimizer" : "adam",
        "learning_rate" : 0.01,
        "momentum" : 0.9,
        "nesterov" : True,
        "activation" : "linear",
        "drop_rate" : 0.3,
        "n_units" : 50,
        "roll2vec" : True,
        "embs_multiplier" : 1,
        "multi_dense" : True,
        "dense_acti" : "linear",
        "full_pred" : True,
    }
    params = merge_params(params, default_params)
    
    docs, target, params_data = data.get_dataset(df_name)
    params = merge_params(params_data, params) # Force the parameters to be the one of the dataset
    X_train, X_test, y_train, y_test = train_test_split(docs, target, test_size=0.3)
    params["split_id"] = random_id() # id to identify the split later
    
    # = = = = = fitting the model on 4 targets = = = = #
    
    # Building the models
    embeddings = data.get_embeddings(roll2vec = params["roll2vec"], multiplier = params["embs_multiplier"])
    print("### EMBS SHAPE : {} ###".format(embeddings.shape))
    model = HAN(embeddings, docs.shape, is_GPU = is_GPU, activation = params["activation"],
                drop_rate=params["drop_rate"], n_units=params["n_units"],
                multi_dense = params["multi_dense"], dense_acti = params["dense_acti"], full_pred = params["full_pred"])
    
    if params["optimizer"]=='sgd':
        decay_rate = params["learning_rate"] / params["nb_epochs"]
        my_optimizer = optimizers.SGD(lr=params["learning_rate"], decay=decay_rate, momentum=params["momentum"], nesterov=params["nesterov"])
    elif params["optimizer"]=='adam':
        my_optimizer = optimizers.Adam()    
    elif params["optimizer"]=='nadam':
        my_optimizer = optimizers.Nadam()
        
    model.compile(loss='mean_squared_error',
                  optimizer=my_optimizer, metrics=['mae'])

    # Training for each target
    params["train_id"] = random_id()
    n_target = 1 if params["full_pred"] else 4
    for tgt in range(n_target):
        t0 = time.process_time()
        # = = = = = training = = = = =
        
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=params["my_patience"],
                                       mode='min')
        
        # save model corresponding to best epoch
        if params["full_pred"]: tgt = "full"
        model_file = os.path.join(data.data_path, "models/", "{}_{}_{}_model.h5".format(model_name, df_name, tgt))
        checkpointer = ModelCheckpoint(filepath=model_file, 
                                       verbose=1, 
                                       save_best_only=True,
                                       save_weights_only=True)
        
        my_callbacks = [early_stopping,checkpointer]
        
        y_train_tgt = y_train if params["full_pred"] else y_train[tgt]
        y_test_tgt = y_test if params["full_pred"] else y_test[tgt]
        model.fit(X_train, 
                  y_train_tgt,
                  batch_size = params["batch_size"],
                  epochs = params["nb_epochs"],
                  validation_data = (X_test,y_test_tgt),
                  callbacks = my_callbacks)
        
        
        T = time.process_time() - t0
        hist = model.history.history
        scores = get_scores(hist)
        scores["T"] = time.process_time() - t0
        
        data.save_perf(params, scores, tgt)
        print("################ {} minutes spent...###########".format(round(T/60)))
              
    return params