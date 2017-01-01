import pickle

'''
Handler of the vocab.txt file  of the form:
dic = {'the': 0, 'a': 1, 'is' : 2, ..., <UNK>: max_vocab_size, <PAD>: max_vocab_size +1 }
where the words are ordered from the most frequent to the last.
'''
class VocabMapping(object):
    def __init__(self, path):
        with open("../data/vocab.txt", "rb") as handle:
            self.dic = pickle.loads(handle.read())

    def getIndex(self, token):
        try:
            return self.dic[token]
        except:
            return self.dic["<UNK>"]

    def getSize(self):
        return len(self.dic)
