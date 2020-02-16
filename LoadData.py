from torch.utils.data import Dataset
import torch

class Dict(object):

    def __init__(self, corpus):
        self.idx2word, self.word2idx = build_dic(corpus)
        self.vocab_size = len(self.word2idx.keys())

    def to_idx(self, word):
        return self.word2idx[word]

    def to_word(self, idx):
        return self.idx2word[idx]

def build_dic(corpus):
    word_count = {}
    with open(corpus, "r") as f:
        for line in f:
            line_split = line.split()
            for word in line_split:
                if word_count.get(word) is None:
                    word_count[word] = 1
                else:
                    word_count[word] = word_count[word] + 1
    count = sorted(word_count.values(), reverse=True)
    print(len(count), "words have been loaded")
    if len(count) > 20000:
        threshold = count[19998]
        word_list = []
        for word in word_count.keys():
            if len(word_list) == 19997:
                break
            if word_count[word] >= threshold:
                word_list.append(word)
        words = word_list
    else:
        words = list(word_count.keys())
    words.append("<unk>")
    words.append("<BOS>")
    words.append("<EOS>")
    assert len(words) == 20000
    idx2word = {}
    word2idx = {}
    for i in range(len(words)):
        idx2word[i] = words[i]
        word2idx[words[i]] = i
    return idx2word, word2idx



class DataSet(Dataset):

    def __init__(self, corpus):
        self.data = open(corpus, 'r').read().strip().split("\n")
        for each in self.data:
            assert len(each.split()) == 5
        self.dic = Dict(corpus)
        self.data_size = len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        split = self.data[item].split()
        replace = []
        for each in split:
            replace.append(each) if each in self.dic.word2idx.keys() else replace.append("<unk>")
        tensor = torch.LongTensor([self.dic.word2idx[replace[0]], self.dic.word2idx[replace[1]],
                  self.dic.word2idx[replace[3]], self.dic.word2idx[replace[4]]])
        return tensor, self.dic.word2idx[replace[2]]


if __name__ == "__main__":

    dic = Dict("data/news.2018.en.shuffled.txt")
    print(dic.vocab_size)
    assert len(dic.word2idx.keys()) == len(dic.idx2word.keys())
