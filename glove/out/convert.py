import numpy as np
import os
import re

if not os.path.exists("temp_vectors.txt"):
	print("Creating new vectors file")
	nf = open("temp_vectors.txt", "w")

	with open("vectors.txt", "r") as f:	
		for line in f.readlines():
			s = re.sub('^(.*?) ',"", line)
			nf.write(s)	 

print("...Converting into npy array")
s = np.loadtxt("temp_vectors.txt")
np.save("vectors.npy", s)
os.remove("temp_vectors.txt")
print("Finished :D")
