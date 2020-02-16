
import torch.nn as nn
import torch


class CBOWNet(nn.Module):

    def __init__(self, vocab_size, dimension):

        super(CBOWNet, self).__init__()
        self.embed = nn.Embedding(vocab_size, dimension)
        self.fc1 = nn.Linear(dimension, vocab_size)
        # self.argmax = torch.argmax(dim=1)
        # self.softmax = nn.LogSoftmax(dim=1)
        # self.fc1 = nn.Linear(vocab_size, dimension)
        # self.fc2 = nn.Linear(dimension, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = x.mean(dim=1)
        # x = self.argmax(x)
        x = self.fc1(x)
        # x = self.softmax(x)
        return x


class SkipGramNet(nn.Module):

    def __init__(self, vocab_size, dimension):

        super(SkipGramNet, self).__init__()
        self.fc1 = nn.Linear(vocab_size, dimension)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):

        x = self.fc1(x)
        x = self.softmax(x)
        return x
