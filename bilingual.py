import torch.utils.data
import torch.optim
import torch.nn as nn
import torch
import argparse
import time
import pickle
import numpy as np
# from scipy.spatial.distance import cdist
import gensim

class Dict(object):

	def __init__(self):
		self.word2idx = {}
		self.idx2word = {}
		self.vocab_size = 0

class Bilingual(nn.Module):

    def __init__(self, en_embed, fr_embed):

        super(Bilingual, self).__init__()
        self.fc = torch.nn.Linear(en_embed, fr_embed)
        self.dropout = torch.nn.Dropout(p=0.4)

    def forward(self, x):
        x = self.fc(x)
        x = self.dropout(x)
        return x


def evaluate(en_embed, fr_embed, network, device):
    total = len(en_test_words)
    correct = 0
    net.eval()

    en_vecs = en_embed[torch.LongTensor(en_test_idxs)].to(device).squeeze()
    en_vecs = network(en_vecs)
    # en_vecs = [en_model[w] for w in en_test_words]
    # en_vecs = torch.FloatTensor(en_vecs).to(device).squeeze()
    # en_vecs = network(en_vecs).squeeze()
    fr_vecs = fr_embed[torch.LongTensor(fr_test_idxs)].to(device).squeeze()
    cos = torch.nn.CosineSimilarity(dim=0)
    highest = -2
    idx = -2
    print("Start to evaluate")
    for j in range(len(en_vecs)):
        for k in range(len(fr_vecs)):
            result = cos(en_vecs[j], fr_vecs[k])
            if result > highest:
                highest = result
                idx = k
        if idx == j:
            print(j, str(correct) + "th correct")
            correct += 1
        highest = -2
        idx = -2


    # distance = cdist(en_vecs, fr_vecs, metric='cosine')
    # distance = np.argmax(distance, axis=0)
    # for j in range(len(distance)):
    #     print(j, distance[j], en_test_words[j], fr_test_words[j], cos(torch.Tensor(en_vecs[j]).squeeze(), torch.Tensor(fr_vecs[j]).squeeze()))
    #     print(j, distance[j], en_test_words[j], fr_test_words[distance[j]], cos(torch.Tensor(en_vecs[j]).squeeze(), torch.Tensor(fr_vecs[distance[j]]).squeeze()))
    #     if distance[j] == j:
    #         correct += 1

    #
    # predict = network(en_vec)
    # similarity = torch.sum(predict.mul(fr_vec), dim=1) / torch.sqrt(torch.sum(predict**2, dim=1))
    # for each in similarity:
    #     if each > 0.5:
    #         correct += 1
    print("Good result percentage:", correct / total)

def orthogonal(matrix):
    u, s, vh = np.linalg.svd(matrix)
    return u.dot(np.eye(matrix.shape[0])).dot(vh.conjugate())


start = time.time()
parser = argparse.ArgumentParser(description='train.py')
parser.add_argument('-epochs', default=500, type=int, help="number of epochs")
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
parser.add_argument('-en_dimension', default=300, type=int, help="embedding dimension of English")
parser.add_argument('-fr_dimension', default=300, type=int, help="embedding dimension of French")
parser.add_argument('-partition_ratio', default=0.7, type=float, help="ratio to partition the data")
en_dic = torch.load("en_vocab")
fr_dic = torch.load("vocab.fr")
language_pairs = pickle.load(open("word_pairs", "rb"))
opt = parser.parse_args()
device = torch.device("cpu")
if len(opt.gpus) != 0:
    torch.cuda.set_device(opt.gpus[0])
    device = torch.device("cuda:" + str(opt.gpus[0]))

en_embed = np.array([torch.load("en_vector_part_1"), torch.load("en_vector_part_2"), torch.load("en_vector_part_3")])
new_en = torch.nn.Embedding(len(en_embed) * en_embed.shape[1], opt.en_dimension)
en_embed = new_en.weight.data.copy_(torch.from_numpy(en_embed).view(-1, opt.en_dimension)).to(device)
# en_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

fr_embed = torch.FloatTensor(torch.load("Word2Vec_CPU.fr")).to(device)

net = Bilingual(opt.en_dimension, opt.fr_dimension).to(device)
criterion = nn.CosineEmbeddingLoss().to(device)
optimizer = torch.optim.Adam(net.parameters())
en_idxs = [en_dic.word2idx[en] for en, _ in language_pairs]
fr_idxs = [fr_dic.word2idx[fr] for _, fr in language_pairs]
assert len(en_idxs) == len(fr_idxs)
en_words = [en for en, _ in language_pairs]
data_len = len(fr_idxs)
print("total data number:", data_len)
en_train_words = en_words[0: int(opt.partition_ratio * data_len)]
en_train_idxs = [en_dic.word2idx[each] for each in en_train_words]
en_test_words = en_words[int(0.7 * data_len): int(1 * data_len)]
en_test_idxs = [en_dic.word2idx[each] for each in en_test_words]
fr_train_idxs = fr_idxs[0: int(opt.partition_ratio * data_len)]
fr_test_idxs = fr_idxs[int(0.7 * data_len): int(1 * data_len)]
# en_test_words = [en_dic.idx2word[each] for each in en_test_idxs]
fr_test_words = [fr_dic.idx2word[each] for each in fr_test_idxs]
# en_train_words = [en_dic.idx2word[each] for each in en_train_idxs]
fr_train_words = [fr_dic.idx2word[each] for each in fr_train_idxs]

net.train()
# en_vec = en_embed[torch.LongTensor(en_train_idxs)].to(device).squeeze()
fr_vec = fr_embed[torch.LongTensor(fr_train_idxs)].to(device).squeeze()
en_vec = en_embed[torch.LongTensor(en_train_idxs)].to(device).squeeze()
# en_vec = [en_model[w] for w in en_train_words]
# en_vec = torch.FloatTensor(en_vec).to(device).squeeze()

for i in range(opt.epochs):
    pred = net(en_vec)
    dot = fr_vec.mul(pred)
    loss = - torch.sum(torch.sum(dot, dim=1) / (torch.sqrt(torch.sum(pred ** 2, dim=1))) / (torch.sqrt(torch.sum(fr_vec ** 2, dim=1))))
    # loss = criterion(pred, fr_vec, target=torch.ones(pred.shape[0]))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(i + 1, "epoch, WXZ:", - loss.item())

list(net.parameters())[0].data.copy_(torch.Tensor(orthogonal(list(net.parameters())[0].data.detach().cpu().numpy()))).to(device)
print("Done orthogonal")
evaluate(en_embed, fr_embed, net, device)

torch.save(net, "bilingual.torch")


