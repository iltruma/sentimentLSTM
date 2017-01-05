'''
This is the main logic for serializing and deserializing dictionaries
of hyperparameters (for use in checkpoint restoration and sampling)
'''
import os
import pickle
import configparser


def read_config_file(path):
    config = configparser.ConfigParser()
    config.read(path)
    return config


def print_parameters(params):
    for p in params:
        print(p + " : " + params[p])


class HyperParameterHandler(object):
    def __init__(self, path):
        self.file_path = os.path.join(path, "hyperparams.p")

    def saveParams(self, dic):
        with open(self.file_path, 'wb') as handle:
            pickle.dump(dic, handle)

    def getParams(self):
        with open(self.file_path, 'rb') as handle:
            return pickle.load(handle)

    def checkExists(self):
        '''
        Checks if hyper parameter file exists
        '''
        return os.path.exists(self.file_path)

    def checkChanged(self, new_params):
        if self.checkExists():
            old_params = self.getParams()
            return old_params["num_layers"] != new_params["num_layers"] or \
                   old_params["hidden_size"] != new_params["hidden_size"] or \
                   old_params["max_seq_length"] != new_params["max_seq_length"] or \
                   old_params["max_vocab_size"] != new_params["max_vocab_size"]
        else:
            return False


def test():
    print_parameters(read_config_file("../config.ini")["sentiment_network_params"])


if __name__ == '__main__':
    test()
