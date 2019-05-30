import os
import keras
import pickle

embedded_data_dir = os.path.join('data','embedded_data.pkl')
vocabulary_embedding_dir = os.path.join('data','vocabulary-embedding.pkl')

maxlendesc = 25
maxlenhead = 25
maxlen = maxlendesc+maxlenhead
rnn_size = 512
rnn_layers = 3
batch_norm = False
activation_rnn_size = 40 if maxlendesc else 0
seed=42
p_W, p_U, p_dense, p_emb, weight_decay = 0, 0, 0, 0, 0
optimizer = 'adam'
learning_rate = 1e-4
batch_size=64
nflips=10
nb_train_samples = 9700
nb_val_samples = 300

with open(vocabulary_embedding_dir, 'rb') as fp:
    embedding, idx2word, word2idx, glove_idx2idx = pickle.load(fp)
vocab_size, embedding_size = embedding.shape

with open(embedded_data_dir, 'rb') as fp:
    X, Y = pickle.load(fp)

nb_unknown_words = 10

print('number of examples',len(X),len(Y))
print('dimension of embedding space for words',embedding_size)
print('vocabulary size', vocab_size, 'the last %d words can be used as place holders for unknown/oov words' % nb_unknown_words)
print('total number of different words',len(idx2word), len(word2idx))
print('number of words outside vocabulary which we can substitue using glove similarity', len(glove_idx2idx))
print('number of words that will be regarded as unknonw(unk)/out-of-vocabulary(oov)',len(idx2word)-vocab_size-len(glove_idx2idx))

for i in range(nb_unknown_words):
    idx2word[vocab_size-1-i] = '<unk%d>' % i
