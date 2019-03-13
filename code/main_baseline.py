import sys
import json
import time
import datetime
import pandas as pd
import numpy as np
import os

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers

# = = = = = = = = = = = = = = =

is_GPU = True
save_weights = True
save_history = True
    




# = = = = = data loading = = = = =

from sklearn.model_selection import train_test_split
import GraphData as data
from HAN import HAN

docs, target = data.get_dataset("small_biased")
X_train, X_test, y_train, y_test = train_test_split(docs, target, test_size=0.2)





# = = = = = fitting the model on 4 targets = = = = =

batch_size = 80
my_patience = 4
embeddings = data.get_embeddings()
res = []
for tgt in range(4):
    model = HAN(embeddings, docs.shape, is_GPU = False)
    
    learning_rate = 0.01
    nb_epochs = 10
    momentum = 0.8
    
    
    decay_rate = learning_rate / nb_epochs
    sgd = optimizers.Adam(lr = learning_rate)
    my_optimizer = sgd # modif 2
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
              y_train,
              batch_size = batch_size,
              epochs = nb_epochs,
              validation_data = (X_test,y_test),
              callbacks = my_callbacks)
    
    hist = model.history.history
    res.append(min(hist["val_loss"]))

    if save_history:
        with open(os.path.join(data.data_path, 'model_history_' + str(tgt) + '.json'), 'w') as file:
            json.dump(hist, file, sort_keys=False, indent=4)




































l_rate, drop, batch_size, n_units = 0.777003, 0.327482, 139, 82
run(l_rate, drop, batch_size, n_units)



# = = = = = Randown Search = = = = =

res = [] # Each run results container
T = 0 # Total time of execution (without cooldown)
for i_run in []:
    # Picking random parameters
    l_rate = np.random.uniform(0.01, 0.8)
    drop = np.random.uniform(0.2, 0.4)
    batch_size = int(np.random.uniform(40, 150))
    n_units = int(np.random.uniform(30, 90))

    # Running the model and storing it
    t0 = time.time()
    print("########## RUN {} for : {} #############".format(i_run, (l_rate, drop, batch_size, n_units)))
    res = run(l_rate, drop, batch_size, n_units)
    t = time.time() - t0
    res = [l_rate, drop, batch_size, n_units, t] + res
    res.append(res)

    # Saving the results at each step
    df = pd.DataFrame(res, columns = ["l_rate", "drop", "batch_size","n_units", "T", "target0", "target1", "target2", "target3"])
    df.to_csv("grid_search.csv")
    T += t
    print("################ {} minutes spent...###########".format(round(T/60)))
    time.sleep(300) # 5 minutes cooldown for the GPU
