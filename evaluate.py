import torch

if __name__ == "__main__":

	word_pair_scores = {}
	cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

	with open("SimLex-999.txt", "r") as f:
		for line in f:
			if not line.startswith("word1\tword2"):
				word1, word2, _, score, _, _, _, _, _, _ = line.split("\t")
				word_pair_scores[(word1, word2)] = float(score)

	vocab = torch.load("vocab")
	Word2Vec = torch.load("Word2Vec_CPU")
	correct = 0
	total = 0

	for word1, word2 in word_pair_scores:
		if word1 in vocab.word2idx and word2 in vocab.word2idx:
			total += 1
			similarity = cos(Word2Vec(torch.LongTensor([vocab.word2idx[word1]])).squeeze(), 
						Word2Vec(torch.LongTensor([vocab.word2idx[word2]])).squeeze())
			if abs(word_pair_scores[(word1, word2)] / 10 - similarity) <= 0.3:
				correct += 1


	print("Total Accuracy:", correct / total)


