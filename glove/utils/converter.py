import numpy as np
import pickle
import os
import re

f = open("../out/vocab.txt", 'r')
nf = open("../../data/vocab.txt", 'wb')

dic = {}
lineno = 0

print("Converting vocab file into binary dictonary...")
with open("../out/vocab.txt", 'r') as f:
    for line in f.readlines():
        s = re.match('^\S*', line).group(0)
        dic[s] = lineno
        lineno += 1
    f.close()

with open("../../data/vocab.txt", 'wb') as nf:
    dic['<UNK>'] = -1
    dic['<PAD>'] = 0
    pickle.dump(dic, nf)
    nf.close()

print("Converting vectors file into npy array...")

with open("../out/temp_vectors.txt", "w") as nf:
	with open("../out/vectors.txt", "r") as f:
		for line in f.readlines():
			s = re.sub('^(.*?) ',"", line)
			nf.write(s)

s = np.loadtxt("../out/temp_vectors.txt")
np.save("../../data/embedding_matrix.npy", s)
os.remove("../out/temp_vectors.txt")
print("Finished :D")
