import json
import os

# = = = = = = = = = = = = = = =


import GraphData as data

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

# = = = = = PREDICTIONS = = = = =     
from HAN import HAN

docs = data.get_kaggle_docs("1000_graphs_not_WL") # Load the raw docs

all_preds_han = []
for tgt in range(4):
    
    print('* * * * * * *',tgt,'* * * * * * *')
    
    model = HAN(data.get_embeddings(), docs.shape, is_GPU = False, activation = "linear")
    
    model.load_weights(data.data_path + 'model_' + str(tgt))
    
    all_preds_han.append(model.predict(docs).tolist())

# flatten
all_preds_han = [elt[0] for sublist in all_preds_han for elt in sublist]

with open(os.path.join(data.data_path, 'predictions_han.txt'), 'w') as file:
    file.write('id,pred\n')
    for idx,pred in enumerate(all_preds_han):
        pred = format(pred, '.7f')
        file.write(str(idx) + ',' + pred + '\n')
