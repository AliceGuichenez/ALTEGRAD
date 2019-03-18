import numpy as np
import pandas as pd

from preprocessing import run_preproc
from training import run_training
from read_results_predict import predictKaggle, show_score


q = 1.2
p = np.random.uniform(min(q,1)-0.3, max(q,1)+0.3)

is_GPU = True

df_name = "unbiased_bigdoc"
model_name = "test"

run_preproc(df_name, N_train = 10000, test = False,
            biased=False, p = None, q = None,
            max_doc_size = 120, num_walks = 10, walk_length=15)

run_training(df_name, model_name,
                 batch_size = 80, my_patience = 2, is_GPU = is_GPU,
                 activation = "linear", nb_epochs = 10, learning_rate = 0.01)


#predictKaggle(df_name, model_name,
#              activation = "linear", is_GPU = is_GPU)


show_score(model_name, df_name)