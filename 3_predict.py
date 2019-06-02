import os
import keras
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import pickle
import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
import random, sys
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.layers.core import Lambda
import keras.backend as K
import h5py
import Levenshtein
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)


embedded_data_dir = os.path.join('data','embedded_data.pkl')
vocabulary_embedding_dir = os.path.join('data','vocabulary-embedding.pkl')


maxlend=25
maxlenh=25
maxlen = maxlend + maxlenh
rnn_size = 512
rnn_layers = 3
batch_norm=False
activation_rnn_size = 40 if maxlend else 0
seed=42
p_W, p_U, p_dense, p_emb, weight_decay = 0, 0, 0, 0, 0
optimizer = 'adam'
batch_size=64
nb_train_samples = 200
nb_val_samples = 20


with open('data/vocabulary-embedding.pkl', 'rb') as fp:
    embedding, idx2word, word2idx, glove_idx2idx = pickle.load(fp)
vocab_size, embedding_size = embedding.shape

with open(embedded_data_dir, 'rb') as fp:
    X, Y = pickle.load(fp)

nb_unknown_words = 10

print('dimension of embedding space for words',embedding_size)
print('vocabulary size', vocab_size, 'the last %d words can be used as place holders for unknown/oov words'%nb_unknown_words)
print('total number of different words',len(idx2word), len(word2idx))
print('number of words outside vocabulary which we can substitue using glove similarity', len(glove_idx2idx))
print('number of words that will be regarded as unknonw(unk)/out-of-vocabulary(oov)',len(idx2word)-vocab_size-len(glove_idx2idx))

for i in range(nb_unknown_words):
    idx2word[vocab_size-1-i] = '<unk%d>'%i
for i in range(vocab_size-nb_unknown_words, len(idx2word)):
    idx2word[i] = idx2word[i]+'^'
empty = 0
eos = 1
idx2word[empty] = '_'
idx2word[eos] = '~'

def prt(label, x):
    print(label+':')
    string_data = ''
    for w in x:
        print(idx2word[w],end=' ')
        string_data += idx2word[w] + ' '
    print()
    return string_data

# region MODEL
# seed weight initialization
random.seed(seed)
np.random.seed(seed)
regularizer = l2(weight_decay) if weight_decay else None
rnn_model = Sequential()
rnn_model.add(Embedding(vocab_size, embedding_size,
                        input_length=maxlen,
                        embeddings_regularizer=regularizer,
                        weights=[embedding], mask_zero=True,
                        name='embedding_1'))
for i in range(rnn_layers):
    lstm = LSTM(rnn_size, dropout=p_W, return_sequences=True,  # batch_norm=batch_norm,
                name='lstm_%d' % (i+1),
                recurrent_regularizer=regularizer,
                kernel_regularizer=regularizer,
                recurrent_dropout=p_U,
                bias_regularizer=regularizer,
                )
    rnn_model.add(lstm)
    rnn_model.add(Dropout(p_dense,name='dropout_%d' % (i+1)))

# endregion

def str_shape(x):
    return 'x'.join(list(map(str,x.shape)))

def inspect_model(model):
    print(model.name)
    for i,l in enumerate(model.layers):
        print(i, 'cls=%s name=%s'%(type(l).__name__, l.name))
        weights = l.get_weights()
        for weight in weights:
            print(str_shape(weight),end=' ')
        print()

def load_weights(model, filepath):
    """Modified version of keras load_weights that loads as much as it can
    if there is a mismatch between file and model. It returns the weights
    of the first layer in which the mismatch has happened
    """
    print('Loading', filepath, 'to', model.name)
    with h5py.File(filepath, mode='r') as f:
        # new file format
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

        # we batch weight value assignments in a single backend call
        # which provides a speedup in TensorFlow.
        weight_value_tuples = []
        for name in layer_names:
            print(name)
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            if len(weight_names):
                weight_values = [g[weight_name] for weight_name in weight_names]
                try:
                    layer = model.get_layer(name=name)
                except(Exception):
                    layer = None
                if not layer:
                    print('failed to find layer', name, 'in model')
                    print('weights', ' '.join(str_shape(w) for w in weight_values))
                    print('stopping to load all other layers')
                    weight_values = [np.array(w) for w in weight_values]
                    break
                symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
                weight_value_tuples += zip(symbolic_weights, weight_values)
                weight_values = None
        K.batch_set_value(weight_value_tuples)
    return weight_values


context_weight = K.variable(1.)
head_weight = K.variable(1.)
cross_weight = K.variable(0.)

def simple_context(X=X, mask=None, n=activation_rnn_size, maxlendesc=maxlend, maxlenhead=maxlenh):
    desc, head = X[:,:maxlendesc,:], X[:,maxlendesc:,:]
    head_activations, head_words = head[:,:,:n], head[:,:,n:]
    desc_activations, desc_words = desc[:,:,:n], desc[:,:,n:]

    # activation for every head word and every desc word
    activation_energies = K.batch_dot(head_activations, desc_activations, axes=(2,2))
    # make sure we dont use description words that are masked out
    # activation_energies = activation_energies + -1e20*K.expand_dims(1.-K.cast(mask[:, :maxlendesc],'float32'),1)

    # for every head word compute weights for every desc word
    activation_energies = K.reshape(activation_energies,(-1,maxlendesc))
    activation_weights = K.softmax(activation_energies)
    activation_weights = K.reshape(activation_weights,(-1,maxlenhead,maxlendesc))

    # for every head word compute weighted average of desc words
    desc_avg_word = K.batch_dot(activation_weights, desc_words, axes=(2,1))
    return K.concatenate((desc_avg_word, head_words))


class SimpleContext(Lambda):
    def __init__(self,**kwargs):
        super(SimpleContext, self).__init__(simple_context,**kwargs)
        self.supports_masking = True

    def compute_mask(self, input, input_mask=None):
        return input_mask[:, maxlend:]

    def get_output_shape_for(self, input_shape):
        nb_samples = input_shape[0]
        n = 2*(rnn_size - activation_rnn_size)
        return (nb_samples, maxlenh, n)


if activation_rnn_size:
    rnn_model.add(SimpleContext(name='simplecontext_1'))
rnn_model.add(TimeDistributed(Dense(vocab_size,
                                    W_regularizer=regularizer, b_regularizer=regularizer,
                                    name='timedistributed_1')))
rnn_model.add(Activation('softmax', name='activation_1'))
weights = load_weights(rnn_model, 'data/model_%03d.hdf5' % 9)

model = Sequential()
model.add(rnn_model)

# if activation_rnn_size:
#     model.add(SimpleContext(name='simplecontext_1'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
n = 2*(rnn_size - activation_rnn_size)
print(n)

inspect_model(model)

# test model

def lpadd(x, maxlend=49, eos=eos):
    """left (pre) pad a description to maxlend and then add eos.
    The eos is the input to predicting the first word in the headline
    """
    assert maxlend >= 0
    if maxlend == 0:
        return [eos]
    n = len(x)
    if n > maxlend:
        x = x[-maxlend:]
        n = maxlend
    return [empty]*(maxlend-n) + x + [eos]

i=3
X_t = np.array(X[i],dtype=np.int)
X_t = list(np.where(X_t>=vocab_size,vocab_size-3,X_t))
print(np.shape(X_t))
samples = [lpadd(X_t)]
# samples = [lpadd([3]*26)]
print(samples)
# pad from right (post) so the first maxlend will be description followed by headline
data = sequence.pad_sequences(samples, maxlen=maxlen, value=empty, padding='post', truncating='post')
print(np.all(data[:,maxlend] == eos))
print(data.shape,list(map(len, samples)))
probs = model.predict([samples], verbose=0, batch_size=1)
print("probs[0].shape",probs[0].shape)
print(probs[0][0])
prt("Input text",X[0])
prt("Predict summary",np.argmax(probs[0],axis=1))
