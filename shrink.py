w = open("large.txt", "w")

maximum = 100000000
count = 0
with open("organised", "r") as f:

	for line in f:
		if line != "\n" and line != "":
			count += 1
			w.write(line)
			if count == maximum:
				break

w.close()

# w = open("large", "w")
# with open("large.txt", "r") as f:
# 	 for line in f:
# 	 	if line != "\n" and line != "":
# 	 		w.write(line)
# w.close()
