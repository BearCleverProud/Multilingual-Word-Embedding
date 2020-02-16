from nltk.corpus import names

names=[name for name in names.words('male.txt')]+[name for name in names.words('female.txt')]

def digit(strings):
	if strings == "." or strings == ",":
		return False
	for each in strings:
		if each not in "0123456789,.":
			return False
	return True


w = open("news.2018.fr.shuffled.tokenized.replace","w")
with open("news.2018.fr.shuffled.tokenized", "r") as f:
	for line in f:
		result = ""
		scripts = line.split(" ")
		for script in scripts:
			if script in names:
				result += "<person> "
			elif digit(script):
				result += "<number> "
			else:
				result += script + " "
		w.write(result.lower())
w.close()