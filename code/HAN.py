import keras

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.models import Model
from keras.layers import Input, Embedding, Dropout, Bidirectional, GRU, CuDNNGRU, TimeDistributed, Dense
from keras import optimizers


class HAN(keras.Model):

    def __init__(self, embeddings, docs_shape, drop_rate=0.1, n_units=50, is_GPU = True, activation = "linear", multi_dense = True, dense_acti = "linear", full_pred = True) :
        sent_ints = Input(shape=(docs_shape[2],))
    
        sent_wv = Embedding(input_dim=embeddings.shape[0],
                            output_dim=embeddings.shape[1],
                            weights=[embeddings],
                            input_length=docs_shape[2],
                            trainable=True,
                            )(sent_ints)
    
        sent_wv_dr = Dropout(drop_rate)(sent_wv)
        sent_wa = bidir_gru(sent_wv_dr,n_units,is_GPU)
        sent_att_vec,word_att_coeffs = AttentionWithContext(return_coefficients=True)(sent_wa)
        sent_att_vec_dr = Dropout(drop_rate)(sent_att_vec)                      
        sent_encoder = Model(sent_ints,sent_att_vec_dr)
        
        # = = = document encoder = = = 
    
        doc_ints = Input(shape=(docs_shape[1],docs_shape[2],))
        sent_att_vecs_dr = TimeDistributed(sent_encoder)(doc_ints)
        doc_sa = bidir_gru(sent_att_vecs_dr,n_units,is_GPU)
        doc_att_vec,sent_att_coeffs = AttentionWithContext(return_coefficients=True)(doc_sa)
        a = Dropout(drop_rate)(doc_att_vec)
        if multi_dense:
            a = Dense(units = 80, activation = dense_acti)(a)
            a = Dense(units = 30, activation = dense_acti)(a)
        h = 4 if full_pred else 1
        preds = Dense(units=h,
                      activation=activation)(a)
        
        return super(HAN, self).__init__(doc_ints, preds, name='HAN')    
                
        



from AttentionWithContext import AttentionWithContext

def bidir_gru(my_seq,n_units,is_GPU):
    '''
    just a convenient wrapper for bidirectional RNN with GRU units
    enables CUDA acceleration on GPU
    # regardless of whether training is done on GPU, model can be loaded on CPU
    # see: https://github.com/keras-team/keras/pull/9112
    '''
    if is_GPU or True:
        return Bidirectional(CuDNNGRU(units=n_units,
                                      return_sequences=True),
                             merge_mode='concat', weights=None)(my_seq)
    else:
        return Bidirectional(GRU(units=n_units,
                                 activation='tanh', 
                                 dropout=0.0,
                                 recurrent_dropout=0.0,
                                 implementation=1,
                                 return_sequences=True,
                                 reset_after=True,
                                 recurrent_activation='sigmoid'),
                             merge_mode='concat', weights=None)(my_seq)