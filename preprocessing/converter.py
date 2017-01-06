import numpy as np
import pickle
import os
import re

'''
Convert glove file for LSTM:
    - convert glove vocabulary into a binary vocabulary for dataprocessor
    - convert glove word vectors into a numpy matrix (vocab_size x dim_vectors) for initial weights
'''


def glove_to_tensorflow(vocab_path, glove_word_vectors_path, outDir):
    vocab_converter(vocab_path, outDir + "glove_vocab.txt")
    embedding_matrix_converter(glove_word_vectors_path, outDir + "embedding_matrix.npy")


def vocab_converter(vocab_in_path, vocab_out_path):
    dic = {}
    lineno = 0

    print("Converting vocab file into binary dictonary...")
    with open(vocab_in_path, 'r') as f:
        for line in f.readlines():
            s = re.match('^\S*', line).group(0)
            dic[s] = lineno
            lineno += 1
        f.close()

    with open(vocab_out_path, 'wb') as nf:
        dic['<UNK>'] = lineno
        dic['<PAD>'] = lineno + 1
        pickle.dump(dic, nf)
        nf.close()


def embedding_matrix_converter(glove_word_vectors_path, embedding_matrix_path):
    print("Converting vectors file into numpy array...")

    with open("/tmp/temp_vectors.txt", "w") as nf:
        with open(glove_word_vectors_path, "r") as f:
            for line in f.readlines():
                s = re.sub('^(.*?) ', "", line)
                nf.write(s)


    n = np.loadtxt("/tmp/temp_vectors.txt")
    pad = np.zeros((1,np.shape(n)[1]))
    nn = np.append(n, pad, axis=0)

    norm_nn = 2.0*(nn - nn.min(axis=0))/(nn.ptp(axis=0))-1.0
    #np.savetxt("../data/norm_vector.txt", norm_nn) #DEBUG

    np.save(embedding_matrix_path, norm_nn)
    os.remove("/tmp/temp_vectors.txt")

    print("Finished :D")

def test(inDir, outDir):
    glove_to_tensorflow(inDir + "vocab.txt", inDir + "vectors.txt", outDir)

if __name__ == '__main__':
    # default directories
    tfDir = "../data/"
    gloveDir = "../data/"
    test(gloveDir, tfDir)
