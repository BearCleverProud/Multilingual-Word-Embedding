import gensim
import torch
import numpy as np

class Dict(object):

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

en_model = gensim.models.KeyedVectors.load_word2vec_format('../../polite-dialogue-generation/data/GoogleNews-vectors-negative300.bin', binary=True)
dict_ = Dict()
vector = []
i = 0
vocab_size = len(en_model.vocab.keys())
for each in en_model.vocab.keys():
	vector.append(en_model[each] / np.sqrt(np.sum(en_model[each] ** 2)))
	dict_.word2idx[each] = i
	dict_.idx2word[i] = each
	i += 1
	if i == vocab_size // 3:
		torch.save(np.array(vector), "en_vector_part_1", pickle_protocol=4)
		vector = []
	elif i == 2 * vocab_size // 3:
		torch.save(np.array(vector), "en_vector_part_2", pickle_protocol=4)
		vector = []
dict_vocab_size = i
torch.save(dict_, "en_vocab")
torch.save(np.array(vector), "en_vector_part_3", pickle_protocol=4)
