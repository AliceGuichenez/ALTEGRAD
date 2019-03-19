import sys
import json
import numpy as np
import os

from keras.models import load_model

# = = = = = = = = = = = = = = =


import GraphData as data
from HAN import HAN


def get_scores(hist):
    ''' Return a dictionary of scores from an history of an epoch'''
    val_mse = hist['val_loss']
    val_mae = hist['val_mean_absolute_error']
    
    min_val_mse = min(val_mse)
    min_val_mae = min(val_mae)
    min_loss = min(hist['loss'])
    
    best_epoch = val_mse.index(min_val_mse) + 1
    
    return {"min_val_mse" : min_val_mse, "min_val_mae" : min_val_mae, "best_epoch" : best_epoch, "min_loss" : min_loss}



# = = = = = PREDICTIONS = = = = =
def predictKaggle(df_name, model_name, params, is_GPU = True):
    '''Use the dataset and the model provided with the params to generate a Kaggle prediction'''
    docs = data.get_kaggle_docs(df_name) # Load the raw docs
    
    all_preds_han = []
    for tgt in range(4):
        
        print('* * * * * * *',tgt,'* * * * * * *')
        
        
        embeddings = data.get_embeddings()
        model = HAN(embeddings, docs.shape, is_GPU = is_GPU, activation = params["activation"], n_units=params["n_units"])
        model_file = os.path.join(data.data_path, "models/", "{}_{}_{}_model.h5".format(model_name, df_name, tgt))
        model.load_weights(model_file)
        all_preds_han.append(model.predict(docs).tolist())
    
    all_preds_han = [elt[0] for sublist in all_preds_han for elt in sublist]
    
    kaggle_file = os.path.join(data.data_path, "predictions/", "preds_{}_{}.txt".format(model_name, df_name))
    with open(kaggle_file, 'w') as file:
        file.write('id,pred\n')
        for idx,pred in enumerate(all_preds_han):
            pred = format(pred, '.7f')
            file.write(str(idx) + ',' + pred + '\n')
    print("The Kaggle file has been saved : {}".format(kaggle_file))            