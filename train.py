import torch.utils.data
import torch.optim
import torch.nn as nn
import torch
import LoadData
import Net
import argparse
import time
import numpy as np

start = time.time()
parser = argparse.ArgumentParser(description='train.py')
parser.add_argument('-epochs', default=100, type=int, help="number of epochs")
parser.add_argument('-batch_size', default=10000, type=int, help="batch size")
parser.add_argument('-embed_dimension', default=200, type=int, help="embedding dimension")
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
parser.add_argument('-corpus', default="../data/small", type=str, help="corpus to train the model")
opt = parser.parse_args()
device = torch.device("cpu")
if len(opt.gpus) != 0:
    torch.cuda.set_device(opt.gpus[0])
    device = torch.device("cuda:" + str(opt.gpus[0]))

data_set = LoadData.DataSet(opt.corpus)
num_data = len(data_set)
print("Training instance:", num_data)
dl = torch.utils.data.DataLoader(data_set, batch_size=opt.batch_size, shuffle=True, num_workers=0)
criterion = nn.CrossEntropyLoss().to(device)
epochs = opt.epochs
network = Net.CBOWNet(20000, opt.embed_dimension).to(device)
print(device)
optimizer = torch.optim.Adam(network.parameters())
cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
best_loss = 1000000.0

for i in range(epochs):
    network.train()
    steps = 0
    for idx, (data, labels) in enumerate(dl):
        data, labels = data.to(device), labels.to(device)
        steps += 1
        predict = network(data)
        loss = criterion(predict, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if steps == 1:
            print("Epoch:", i+1, "Steps:", steps, "PPL:", 2 ** loss.item())
        if loss.item() < best_loss:
            weight = list(network.embed.parameters())[0].data.cpu().numpy()
            new_weight = np.zeros_like(weight)
            for j in range(len(weight)):
                sum = np.sqrt(np.sum(np.square(weight[j])))
                new_weight[j] = weight[j] / sum
            network.embed.weight.data.copy_(torch.from_numpy(new_weight))
            best_loss = loss.item()
            torch.save(network.embed, "Word2Vec.fr")
            torch.save(network.embed.cpu(), "Word2Vec_CPU.fr")
            torch.save(network.embed.weight.cpu().detach().numpy(), "numpy_embed_test.fr")
            torch.save(data_set.dic, "vocab.fr")
            embed = torch.load("Word2Vec.fr")
            dic = torch.load("vocab.fr")
            king = torch.LongTensor([dic.word2idx["roi"]]).to(device)
            man = torch.LongTensor([dic.word2idx["homme"]]).to(device)
            woman = torch.LongTensor([dic.word2idx["femme"]]).to(device)
            queen = torch.LongTensor([dic.word2idx["reine"]]).to(device)
            king_minus_man_plus_woman = embed(king) - embed(man) + embed(woman)
            queen = embed(queen)
            output = "Cosine similarity between \"king + man - woman\" and \"queen\": {0:.3f}"
            print(output.format(cos(king_minus_man_plus_woman.squeeze(), queen.squeeze()).item()))
            paris = torch.LongTensor([dic.word2idx["paris"]]).to(device)
            france = torch.LongTensor([dic.word2idx["france"]]).to(device)
            uk = torch.LongTensor([dic.word2idx["royaume-uni"]]).to(device)
            london = torch.LongTensor([dic.word2idx["londres"]]).to(device)
            paris_minus_france_plus_london = embed(paris) - embed(france) + embed(uk)
            london = embed(london)
            output = "Cosine similarity between \"paris - france + uk\" and \"london\": {0:.3f}"
            print(output.format(cos(paris_minus_france_plus_london.squeeze(), london.squeeze()).item()))
            network = network.to(device)
