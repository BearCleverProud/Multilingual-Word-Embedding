
w = open("small.fr", "w")
enough = False
count = 0
maximum = 1000000
with open("news.2018.fr.shuffled.tokenized.replace", "r") as f:

	for line in f:
		line_split = line.split()
		line_split.insert(0, "<bos>")
		line_split.insert(0, "<bos>")
		line_split.append("<eos>")
		line_split.append("<eos>")
		for i in range(2, len(line_split) - 2):
			new_line = line_split[i - 2] + " " + line_split[i - 1] + " " + line_split[i] + " " + line_split[i + 1] + " " + line_split[i + 2] + "\n"
			count += 1
			if count % 10000 == 0:
				print(count)
			if count == maximum:
				enough = True
				break
			else:
				w.write(new_line)
		if enough:
			break

w.close()