import numpy as np
import pandas as pd

from preprocessing import run_preproc
from training import run_training
from scores_kaggle import predictKaggle

import GraphData as data

import random, time



# Generation parameters
df_name = "test"
model_name = "test"
run_Kaggle = False
is_GPU = True


# = = Hyper-parameters = = #
# If not set it will use the default in the function
# Thus it is incremented through the execution
# It is very useful to track our hyper parameter through the execution
#  and save it with its performance.
params = {
    "N_train" : 60000 if not(run_Kaggle) else None,
    "biased" : True,
    "activation" : "linear",
    "optimizer" : "adam",
    "nb_epochs" : 10,
}

for i in range(60):
    print("#### RUN {} ####".format(i+1))
    params["p"] = random.uniform(0.5,1.5)
    params["q"] = random.uniform(0.5,1.5)

    params = run_preproc(df_name, test = run_Kaggle, params = params)
    
    params = run_training(df_name, model_name, is_GPU = is_GPU, params = params)
    
    if run_Kaggle:
        predictKaggle(df_name, model_name, is_GPU = is_GPU, params = params)
        
    
    perfs = data.get_perfs(params["train_id"])
    print(perfs)
    
    time.sleep(30) # Cooldown