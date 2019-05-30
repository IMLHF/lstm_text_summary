import jsonlines
import os
from collections import Counter
from itertools import chain
import matplotlib.pyplot as plt
import subprocess
import numpy as np
import string
import pickle

# region get word2idx[word] and idx2word[idx]
all_data_f = open(os.path.join("data","sample-1M.jsonl"),'r')
data_row = 0
heads = []
desc = []
max_data_row = 10000
for item in jsonlines.Reader(all_data_f):
  data_row += 1
  title = item['title']
  content = item['content']
  for p in string.punctuation:
    title = title.replace(p," "+p+" ")
    content = content.replace(p," "+p+" ")
  heads.append(title)
  desc.append(content)
  if data_row >= max_data_row:
    break
print('data_row: %d.' % len(heads))
# print(heads[0])
# print(desc[0]+'233333333333333333')

def get_vocab(lst):
  '''
  lst : all sentence list to extract all word.
  '''
  vocabcount = Counter(w for txt in lst for w in txt.split()) # all word
  vocab = list(map(lambda x: x[0], sorted(vocabcount.items(), key=lambda x: -x[1]))) # popular descending sort
  print("all word num: %d." % len(vocab))
  return vocab, vocabcount

vocab, vocabcount = get_vocab(heads+desc)

# print(vocabcount[vocab[0]],vocabcount[vocab[1]],vocabcount[vocab[2]],)
# print(vocab[:50], len(vocab))
# plt.plot([vocabcount[w] for w in vocab])
# plt.gca().set_xscale("log", nonposx='clip')
# plt.gca().set_yscale("log", nonposy='clip')
# plt.title('word distribution in headlines and discription')
# plt.xlabel('rank')
# plt.ylabel('total appearances')
# plt.show()

def get_idx(vocab, vocabcount):
  empty = 0 # RNN mask of no data
  eos = 1  # end of sentence
  start_idx = eos+1 # first real word
  word2idx = dict((word, idx+start_idx) for idx, word in enumerate(vocab))
  word2idx['<empty>'] = empty
  word2idx['<eos>'] = eos
  idx2word = dict((idx, word) for word, idx in word2idx.items())
  return word2idx, idx2word

word2idx, idx2word = get_idx(vocab, vocabcount)
# endregion

# region read glove Word Embedding (glove_index_dict and glove_embedding_weight)
'''
glove_index_dict[word]=index
glove_embedding_weight[index][:] = word_embedding_vec
'''
embedding_dim = 100
fname = 'glove.6B.%dd.txt' % embedding_dim
glove_name = os.path.join('data', fname)
if not os.path.exists(glove_name):
  print('glove file not exist, please download at "http://nlp.stanford.edu/data/glove.6B.zip".')
  exit(-1)
  # path = 'glove.6B.zip'
  # path = get_file(path, origin="http://nlp.stanford.edu/data/glove.6B.zip")
  # !unzip {datadir}/{path}
state, result = subprocess.getstatusoutput('wc -l '+glove_name)
if state != 0:
  print('state error, state:%d.' % state)
  exit(-1)
# print(result)
glove_n_symbols = int(result.split()[0]) # glove中单词的数量
print('glove_n_symbols: %d.' % glove_n_symbols)
glove_index_dict = {} # glove中单词到编号的映射
glove_embedding_weights = np.empty((glove_n_symbols, embedding_dim),
                                   dtype=np.float32)  # glove中单词编号到embedding矩阵的映射
global_scale = .1
with open(glove_name,'r') as fp:
  i=0
  for line in fp:
    line = line.strip().split()
    word = line[0]
    glove_index_dict[word]=i
    glove_embedding_weights[i] = np.array(line[1:], dtype=np.float32)
    i += 1
glove_embedding_weights *= global_scale
glove_embedding_weights_std = glove_embedding_weights.std()
print('glove_embedding_weights_std: %f.' % glove_embedding_weights_std)

# 检查纯小写的单词是否在dict中，不在则将其映射到相应地单词上（忽略大小写）。
for w,i in glove_index_dict.items():
    w = w.lower()
    if w not in glove_index_dict:
        glove_index_dict[w] = i
# endregion

# region embedding matrix
seed = 42
vocab_size = 40000
np.random.seed(seed)
shape = (vocab_size, embedding_dim)
scale = glove_embedding_weights_std*np.sqrt(12)/2
embedding = np.random.uniform(low=-scale,high=scale,size=shape)
print('random-embedding/glove scale', scale, 'std', embedding.std())

# copy from glove weights of words that appear in our short vocabulary (idx2word)
c = 0
for i in range(vocab_size):
  word = idx2word[i]
  glove_embedding_idx = glove_index_dict.get(word, glove_index_dict.get(word.lower()))
  if glove_embedding_idx is not None:
    embedding[i] = glove_embedding_weights[glove_embedding_idx]
    c+=1
  else:
    '''
    how to process word like "trillion,".
    '''
    pass
    # print(word)
print('%d (all %d) tokens found in glove and copied to embedding.' % (c, vocab_size))
# endregin

# region map word out of embedding_matrix to similar word or <unknown>
glove_thr = 0.5
word2glove = {}
for w in word2idx:
    if w in glove_index_dict:
        g = w
    elif w.lower() in glove_index_dict:
        g = w.lower()
    elif w.startswith('#') and w[1:] in glove_index_dict:
        g = w[1:]
    elif w.startswith('#') and w[1:].lower() in glove_index_dict:
        g = w[1:].lower()
    else:
        continue
    word2glove[w] = g

normed_embedding = embedding/np.array([np.sqrt(np.dot(gweight,gweight)) for gweight in embedding])[:,None]

nb_unknown_words = 100

glove_match = []
for w,idx in word2idx.items():
    if idx >= vocab_size-nb_unknown_words and w.isalpha() and w in word2glove:
        gidx = glove_index_dict[word2glove[w]]
        gweight = glove_embedding_weights[gidx,:].copy()
        # find row in embedding that has the highest cos score with gweight
        gweight /= np.sqrt(np.dot(gweight,gweight))
        score = np.dot(normed_embedding[:vocab_size-nb_unknown_words], gweight)
        while True:
            embedding_idx = score.argmax()
            s = score[embedding_idx]
            if s < glove_thr:
                break
            if idx2word[embedding_idx] in word2glove:
                glove_match.append((w, embedding_idx, s))
                break
            score[embedding_idx] = -1
glove_match.sort(key=lambda x: -x[2])
print('# of glove substitutes found', len(glove_match))
for orig, sub, score in glove_match[-10:]:
    print(score, orig,'=>', idx2word[sub])
glove_idx2idx = dict((word2idx[w], embedding_idx)
                     for w, embedding_idx, _ in glove_match)
# endregion



X = [[word2idx[token] for token in d.split()] for d in desc]
# len(X)
# plt.hist(list(map(len,X)),bins=50)
# plt.show()
Y = [[word2idx[token] for token in headline.split()] for headline in heads]
# print(len(Y))
# print(Y[0])
# plt.hist(list(map(len,Y)),bins=50)
# plt.show()

with open('data/%s.pkl' % "vocabulary-embedding",'wb') as fp:
    pickle.dump((embedding, idx2word, word2idx, glove_idx2idx),fp,-1)

with open('data/embedded_data.pkl','wb') as fp:
    pickle.dump((X,Y),fp,-1)
