import os
import pickle
import re


f = open("../out/vocab.txt", 'r')
nf = open("../../data/vocab_glove.txt", 'wb')

dic = {}
lineno = 0

print("Creating new vocab file")
for line in f.readlines():
    s = re.match('^\S*', line).group(0)
    dic[s] = lineno
    lineno += 1

pickle.dump(dic, nf)
print("Finished :D")
