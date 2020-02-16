import io
import torch
import numpy as np

class Dict(object):

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    vocab = Dict()
    vector = []
    i = 0
    for line in fin:
        if i == 0:
            i += 1
            continue
        tokens = line.rstrip().split(' ')
        fr, vecs = tokens[0], tokens[1:]
        vec = np.array([float(component) for component in vecs])
        vec /= np.sqrt(np.sum(vec ** 2))
        vec = vec.tolist()
        vector.append(vec)
        vocab.word2idx[fr] = i - 1
        vocab.idx2word[i - 1] = fr
        vocab.vocab_size = i
        i += 1
    return vocab, vector

if __name__ == '__main__':
    vocab, vector = load_vectors("cc.fr.300.vec")
    vector = torch.save(torch.Tensor(vector), "Word2Vec_CPU.fr")
    torch.save(vocab, "vocab.fr")
