import pickle
import re

'''
Handler of the vocab.txt file  of the form:
dic = {'the': 0, 'a': 1, 'is' : 2, ..., <UNK>: max_vocab_size, <PAD>: max_vocab_size +1 }
where the words are ordered from the most frequent to the last.
'''
class VocabMapping(object):
    def __init__(self, path, glove=False):
        with open(path, "rb") as handle:
            if glove:
                dic = {}
                line_num = 0

                print("Converting vocab file into binary dictonary...")
                for line in handle.readlines():
                    s = re.match('^\S*', line).group(0)
                    dic[s] = line_num
                    line_num += 1


                dic['<UNK>'] = line_num
                dic['<PAD>'] = line_num + 1
                self.dic = dic
            else:
                self.dic = pickle.loads(handle.read())

    def getIndex(self, token):
        try:
            return self.dic[token]
        except:
            return self.dic["<UNK>"]

    def getSize(self):
        return len(self.dic)


def test():
    voc = VocabMapping("../data/vocab.txt")
    print("size: {}, index of 'hello': {}".format(voc.getSize(), voc.getIndex("hello")))
    print(voc.dic)


if __name__ == '__main__':
    test()