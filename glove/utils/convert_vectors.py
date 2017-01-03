import numpy as np
import os
import re

if not os.path.exists("../out/temp_vectors.txt"):
	print("Creating new vectors file")
	nf = open("../out/temp_vectors.txt", "w")

	with open("../out/vectors.txt", "r") as f:
		for line in f.readlines():
			s = re.sub('^(.*?) ',"", line)
			nf.write(s)

print("...Converting into npy array")
s = np.loadtxt("../out/temp_vectors.txt")
np.save("../out/vectors.npy", s)
os.remove("../out/temp_vectors.txt")
print("Finished :D")
