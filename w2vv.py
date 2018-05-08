import numpy as np 
np.random.seed(1337)

from keras.layers import Dense, Dropout, Input, concatenate, Embedding, GRU, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop, Adam
from keras.regularizers import l2
from keras.utils import plot_model
from keras.models import Model, Sequential

import keras.backend as K

# basic word2visualvec
class Base_model:
    def compile_model(self, loss_name, opt=None):
        print "loss function: ", loss_name
        print "optimizer: ", opt.optimizer
        print "learning_rate: ", opt.lr
        if loss_name == 'mse':
            loss = loss_name

        clipnorm = opt.clipnorm
        optimizer = opt.optimizer
        learning_rate = opt.lr
        if optimizer == 'sgd':
            # let's train the model using SGD + momentum (how original).
            if clipnorm > 0:
                sgd = SGD(lr=learning_rate, clipnorm=clipnorm, decay=1e-6, momentum=0.9, nesterov=True)
            else:
                sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
            self.model.compile(loss=loss, optimizer=sgd)
        elif optimizer == 'rmsprop':
            if clipnorm > 0:
                rmsprop = RMSprop(lr=learning_rate, clipnorm=clipnorm, rho=0.9, epsilon=1e-6)
            else:
                rmsprop = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-6)
            self.model.compile(loss=loss, optimizer=rmsprop)
        elif optimizer == 'adagrad':
            if clipnorm > 0:
                adagrad = Adagrad(lr=learning_rate, clipnorm=clipnorm, epsilon=1e-06)
            else:
                adagrad = Adagrad(lr=learning_rate, epsilon=1e-06)
            self.model.compile(loss=loss, optimizer=adagrad)
        elif optimizer == 'adma':
            if clipnorm > 0:
                adma = Adam(lr=learning_rate, clipnorm=clipnorm, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            else:
                adma = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            self.model.compile(loss=loss, optimizer=adma) 
            
    def init_model(self, fname):
        self.model.load_weights(fname)


    def save_json_model(self, model_file_name):
        json_string = self.model.to_json()
        if model_file_name[-5:] != '.json':
            model_file_name = model_file_name + '.json'
        open(model_file_name, 'w').write(json_string)

    def plot(self, filename):
        plot_model(self.model, to_file=filename, show_shapes=True, show_layer_names=True)

    def get_lr(self):
        #return self.model.optimizer.lr.get_value()
        return K.get_value(self.model.optimizer.lr)

    def decay_lr(self, decay=0.9):
        old_lr = self.get_lr()
        new_lr = old_lr * decay
        # new_lr = old_lr / (1 + decay*epoch)
        K.set_value(self.model.optimizer.lr, new_lr)




class W2VV_MS(Base_model):
    def __init__(self, opt, we_weights=None):

        n_layers = opt.n_text_layers

        assert (opt.rnn_style in ['lstm', 'gru']), "not supported LSTM style (%s)" % lstm_style

        # creat model
        print("Building model...")

        # input word index for rnn
        main_input = Input(shape=(opt.sent_maxlen,))
        # bow, word2vec or word hashing embedded sentence vector
        auxiliary_input = Input(shape=(n_layers[0],))

        if we_weights is None:
            we = Embedding(opt.vocab_size, opt.embed_size)(main_input)
        else:
            we = Embedding(opt.vocab_size, opt.embed_size, trainable=True, weights = [we_weights])(main_input)
        we_dropout = Dropout(opt.dropout)(we)

        #lstm_out = LSTM(lstm_size, return_sequences=False, unroll=True, consume_less='gpu', init='glorot_uniform')(we_dropout)
        if opt.rnn_style == 'lstm':
            lstm_out = LSTM(opt.rnn_size, return_sequences=False, unroll=True, dropout=opt.dropout, recurrent_dropout=opt.dropout)(we_dropout)
        elif opt.rnn_style == 'gru':
            lstm_out = GRU(opt.rnn_size, return_sequences=False, unroll=True, dropout=opt.dropout, recurrent_dropout=opt.dropout)(we_dropout)
        
        x = concatenate([lstm_out, auxiliary_input], axis=-1)
        for n_neuron in range(1,len(n_layers)-1):
            x = Dense(n_layers[n_neuron], activation=opt.hidden_act, kernel_regularizer=l2(opt.l2_p))(x)
            x = Dropout(opt.dropout)(x)
        
        output = Dense(n_layers[-1], activation=opt.hidden_act, kernel_regularizer=l2(opt.l2_p))(x)

        self.model = Model(inputs=[main_input, auxiliary_input], outputs=output)
        self.model.summary()


    def predict_one(self, text_vec, text_vec_2):
        text_embed_vec = self.model.predict([np.array([text_vec]), np.array([text_vec_2])])
        return text_embed_vec[0]

    def predict_batch(self, text_vec_batch, text_vec_batch_2):
        text_embed_vecs = self.model.predict([np.array(text_vec_batch), np.array(text_vec_batch_2)])
        return text_embed_vecs



NAME_TO_MODEL = {'w2vv_ms': W2VV_MS}


def get_model(name):
    return NAME_TO_MODEL[name]