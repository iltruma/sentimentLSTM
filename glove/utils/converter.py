import numpy as np
import pickle
import os
import re

# default directories
dataDir = "../../data/"
outDir = "../out/"

'''
Convert glove file for LSTM:
    - convert glove vocabulary into a binary vocabulary for dataprocessor
    - convert glove word vectors into a numpy matrix (vocab_size x dim_vectors) for initial weights
'''


def glove_converter(dataDir, outDir):
    dic = {}
    lineno = 0

    print("Converting vocab file into binary dictonary...")
    with open(outDir + "vocab.txt", 'r') as f:
        for line in f.readlines():
            s = re.match('^\S*', line).group(0)
            dic[s] = lineno
            lineno += 1
        f.close()

    with open(dataDir + "vocab.txt", 'wb') as nf:
        dic['<UNK>'] = lineno
        dic['<PAD>'] = lineno + 1
        pickle.dump(dic, nf)
        nf.close()

    print("Converting vectors file into npy array...")

    with open(outDir + "temp_vectors.txt", "w") as nf:
        with open(outDir + "vectors.txt", "r") as f:
            for line in f.readlines():
                s = re.sub('^(.*?) ', "", line)
                nf.write(s)

        nf.close()
        f.close()

    n = np.loadtxt(outDir + "temp_vectors.txt")
    np.save(dataDir + "embedding_matrix.npy", n)
    os.remove(outDir + "temp_vectors.txt")

    print("Finished :D")


if __name__ == '__main__':
    glove_converter(dataDir, outDir)
