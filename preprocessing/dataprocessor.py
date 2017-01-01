'''
Process the dataset to extract the vocabulary (dataDir + "vocab.txt") and the
processed data (dataDir + "processed") as described by the following comments
'''
import os
import nltk
import pickle
import urllib
import numpy as np
from multiprocessing import Process, Lock

dataDir = "../data/"
dirs = [dataDir + "aclImdb/test/pos", dataDir + "aclImdb/test/neg", \
        dataDir + "aclImdb/train/pos", dataDir + "aclImdb/train/neg"]
url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

'''
checks if files and directories for the data are already present, if not makes them

To speed up the data processing (I probably did it way too inefficiently),
I decided to split the task in n processes, where n is the number of directories
Each of the four (test/pos, test/neg, train/pos, train/neg) directory are processed in the same way specified below
'''


def run(max_seq_length, max_vocab_size, min_count):
    if not os.path.exists(dataDir):
        print('%s directory not found!' % dataDir)
        return
    if not os.path.exists(dataDir + "checkpoints/"):
        os.makedirs(dataDir + "checkpoints")
    if not os.path.isdir(dataDir + "aclImdb"):
        print("Data not found, downloading dataset...")
        fileName = downloadFile(url)
        import tarfile
        tfile = tarfile.open(fileName, 'r:gz')
        print("Extracting dataset...")
        tfile.extractall(dataDir)
        tfile.close()
    if os.path.exists(dataDir + "vocab.txt"):
        print("vocab mapping found...")
    else:
        print("no vocab mapping found, running preprocessor...")
        createVocab(dirs, max_vocab_size, min_count)
    if not os.path.exists(dataDir + "processed"):
        os.makedirs(dataDir + "processed/")
        print("No processed data files found, running preprocessor...")
    else:
        print("Processed data files found: delete " + dataDir + "processed  to redo them")
        return
    import vocabmapping
    vocab = vocabmapping.VocabMapping(dataDir + "vocab.txt")
    dirCount = 0
    processes = []
    lock = Lock()
    for d in dirs:
        print("Procesing data with process: " + str(dirCount))
        p = Process(target=createProcessedDataFile, args=(vocab, d, dirCount, max_seq_length, lock))
        p.start()
        processes.append(p)
        dirCount += 1
    for p in processes:
        if p.is_alive():
            p.join()


'''
Multithread implementation of the data processing (a thread per directory)
A lock was used to ensure while writing to std.out bad things don't happen.


- for each file in the directory:
    - tokenize the lowercase text review as one sentence: (standard tokenizer of the nltk library)
       - split standard contractions, e.g. ``don't`` -> ``do n't`` and ``they'll`` -> ``they 'll``
       - treat most punctuation characters as separate tokens
       - split off commas and single quotes, when followed by whitespace
       - separate periods that appear at the end of line
       - etc...
    - find the index for each token based on the vocabulary mapping
    - save indices on a numpy array of dimension max_seq_length + 2 where the last 2 values are
      one (0/1) for the positive or negative review and the other for the review length:
      if the review has length less than max_seq_length the remaining index are for the '<PAD>' token
      if there are words not present in the vocabulary the index of the '<UNK>' token is used
- stack all the numpy arrays vertically (as raw of a numpy matrix) and save them to memory
'''


def createProcessedDataFile(vocab_mapping, directory, pid, max_seq_length, lock):
    count = 0
    data = np.array([i for i in range(max_seq_length + 2)])
    for f in os.listdir(directory):
        count += 1
        if count % 100 == 0:
            lock.acquire()
            print("Processing: " + f + " the " + str(count) + "th file... on process: " + str(pid))
            lock.release()
        with open(os.path.join(directory, f), 'r') as review:
            tokens = tokenize(review.read().lower())
            numTokens = len(tokens)
            indices = [vocab_mapping.getIndex(j) for j in tokens]
            # pad sequence to max length
            if len(indices) < max_seq_length:
                indices = indices + [vocab_mapping.getIndex("<PAD>") for i in range(max_seq_length - len(indices))]
            else:
                indices = indices[0:max_seq_length]
        if "pos" in directory:
            indices.append(1)
        else:
            indices.append(0)
        indices.append(min(numTokens, max_seq_length))
        assert len(indices) == max_seq_length + 2, str(len(indices))
        data = np.vstack((data, indices))
        indices = []
    # remove first placeholder value
    data = data[1::]
    lock.acquire()
    print("Saving data file{0} to disk...".format(str(pid)))
    lock.release()
    saveData(data, pid)


# method from: http://stackoverflow.com/questions/22676/how-do-i-download-a-file-over-http-using-python
def downloadFile(url):
    file_name = os.path.join(dataDir, url.split('/')[-1])
    u = urllib.request.urlopen(url)
    f = open(file_name, 'wb')
    meta = u.info()
    file_size = int(meta.get_all("Content-Length")[0])
    print("Downloading: %s Bytes: %s" % (file_name, file_size))
    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break
        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        print(status)
    f.close()
    return file_name


'''
This function tokenizes sentences
'''


def tokenize(text):
    return nltk.word_tokenize(text)


'''
Saves processed data numpy array
'''


def saveData(npArray, index):
    name = "data{0}.npy".format(str(index))
    outfile = os.path.join(dataDir + "processed/", name)
    print("numpy array is: {0}x{1}".format(len(npArray), len(npArray[0])))
    np.save(outfile, npArray)


'''
create vocab mapping file and stores it in vocab.txt (this is not a text file but a binary):
the data is of the form dic = {'the': 0, 'a': 1, 'is' : 2, ..., <UNK>: max_vocab_size, <PAD>: max_vocab_size +1 }
where the words are ordered from the most frequent to the last.
'''


def createVocab(dirs, max_vocab_size, min_count):
    print("Creating vocab mapping (max size: %d, min frequency: %d)..." %(max_vocab_size, min_count))
    dic = {}
    for d in dirs:
        indices = []
        for f in os.listdir(d):
            with open(os.path.join(d, f), 'r') as review:
                tokens = tokenize(review.read().lower())
                for t in tokens:
                    if t not in dic:
                        dic[t] = 1
                    else:
                        dic[t] += 1
    d = {}
    counter = 0
    for w in sorted(dic, key=dic.get, reverse=True):
        # take word more frequent than min_count
        if dic[w] < min_count: break
        d[w] = counter
        counter += 1
        # take most frequent max_vocab_size tokens
        if max_vocab_size > -1 and counter >= max_vocab_size: break

    # add out of vocab token and pad token
    d["<UNK>"] = counter
    counter += 1
    d["<PAD>"] = counter
    print("vocab mapping created: size: %d discarded: %d" % (len(d), len(dic) - len(d) + 2 ))
    with open(dataDir + 'vocab.txt', 'wb') as handle:
        pickle.dump(d, handle)


def main():
    max_seq_length = 200 # max sequence dimension
    max_vocab_size = -1 # max vocabulary dimension (-1 for total dimension) (20000)
    min_count = 5 # discard word with low frequency
    run(max_seq_length, max_vocab_size, min_count)


if __name__ == '__main__':
    main()