import pickle
import torch
import LoadData
import string
import gensim

class Dict(object):

	def __init__(self):
		self.word2idx = {}
		self.idx2word = {}
		self.vocab_size = 0

def extract_words(en2frfile, fr2enfile):

	en_dict = {}
	fr_dict = {}
	threshold = 0.05

	with open(en2frfile, "r") as f:
		for line in f:
			fr, en, prob = line.split()
			if float(prob) > threshold:
				fr_dict[(fr, en)] = float(prob)

	print("Done loading FR to EN dictionary,", len(fr_dict.keys()), "pairs have been loaded")

	with open(fr2enfile, "r") as f:
		for line in f:
			en, fr, prob = line.split()
			if float(prob) > threshold:
				en_dict[(en, fr)] = float(prob)

	print("Done loading EN to FR dictionary,", len(en_dict.keys()), "pairs have been loaded")

	return en_dict, fr_dict

if __name__ == "__main__":
	# en_dict, fr_dict = extract_words("lex.1.e2f", "lex.1.f2e")
	# pickle.dump(en_dict, open("en_dict", "wb"))
	# print("Done dumped EN dictionary")
	# pickle.dump(fr_dict, open("fr_dict", "wb"))
	# print("Done dumped FR dictionary")
	
	en_dict = pickle.load(open("en_dict", "rb"))
	print("Done loaded en_dict")
	fr_dict = pickle.load(open("fr_dict", "rb"))
	print("Done loaded fr_dict")
	en_vocab = gensim.models.KeyedVectors.load_word2vec_format("../../polite-dialogue-generation/data/GoogleNews-vectors-negative300.bin", binary=True).vocab.keys()
	fr_vocab = torch.load("../code/vocab.fr")
	pairs = []
	for en, fr in en_dict.keys():
		if (len(en) == 1 and en in string.punctuation) or (len(fr) == 1 and fr in string.punctuation) or en == fr:
			continue
		if fr_dict.get((fr, en)) is not None and en in en_vocab and fr in fr_vocab.word2idx.keys() and en_dict[(en, fr)] * fr_dict[(fr, en)] > 0.1:
			print(en, fr, en_dict[(en, fr)], fr_dict[(fr, en)])
			pairs.append((en, fr))

	print(len(pairs), "of word pairs have been loaded.")
	pickle.dump(pairs, open("word_pairs", "wb"))
	print("Done dumped word pairs")





