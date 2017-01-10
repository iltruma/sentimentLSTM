import os
import numpy as np


class DataHandler(object):
    def __init__(self, data_path, batch_size, max_examples=-1, shuffle_each_pass=True, train_seed=None):
        print("init DataHandler with path:{}, batch_size:{}".format(data_path,batch_size) + ", max number of examples:{}".format(max_examples))
        self.batch_size = batch_size
        self.shuffle_each_pass = shuffle_each_pass

        train_path = data_path + "train/"
        test_path =  data_path + "test/"
        # input files (numpy matrices, 1 row per example)
        trainFiles = [ f for f in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, f))]
        testFiles = [ f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))]

        # load all files in memory
        self.train_data = np.load(os.path.join(train_path, trainFiles[0]))
        self.test_data = np.load(os.path.join(test_path, testFiles[0]))

        for i in range(1, len(trainFiles)):
            self.train_data = np.vstack((self.train_data, np.load(os.path.join(train_path, trainFiles[i]))))

        for i in range(1, len(testFiles)):
            self.test_data = np.vstack((self.test_data, np.load(os.path.join(test_path, testFiles[i]))))

        # randomize the dataset
        np.random.shuffle(self.train_data)
        np.random.shuffle(self.test_data)

        # use only max_example examples when defined
        if max_examples >-1 and max_examples < len(self.train_data) and max_examples < len(self.test_data):
            self.train_data = self.train_data[:max_examples]
            self.test_data = self.test_data[:max_examples]

        self.train_batch_pointer = 0
        self.test_batch_pointer = 0

        tr_targets = (self.train_data.transpose()[-2]).transpose()
        tr_onehot = np.zeros((len(tr_targets), 2))
        tr_onehot[np.arange(len(tr_targets)), tr_targets] = 1
        tr_sequence_lengths = (self.train_data.transpose()[-1]).transpose()
        self.train_data = (self.train_data.transpose()[0:-2]).transpose()

        te_targets = (self.test_data.transpose()[-2]).transpose()
        te_onehot = np.zeros((len(te_targets), 2))
        te_onehot[np.arange(len(te_targets)), te_targets] = 1
        te_sequence_lengths = (self.test_data.transpose()[-1]).transpose()
        self.test_data = (self.test_data.transpose()[0:-2]).transpose()


        # cutoff non even number of batches (is it necessary?)
        num_train_batches = len(self.train_data) // self.batch_size
        num_test_batches = len(self.test_data) // self.batch_size
        train_cutoff = len(self.train_data) - (len(self.train_data) % self.batch_size)
        test_cutoff = len(self.test_data) - (len(self.test_data) % self.batch_size)
        self.train_data = self.train_data[:train_cutoff]
        self.test_data = self.test_data[:test_cutoff]

        print(" Train size is: {0}, splitting into {1} batches".format(len(self.train_data), num_train_batches))

        self.train_sequence_lengths = tr_sequence_lengths[:train_cutoff]
        self.train_sequence_lengths = np.split(self.train_sequence_lengths, num_train_batches)
        self.train_targets = tr_onehot[:train_cutoff]
        self.train_targets = np.split(self.train_targets, num_train_batches)
        self.train_data = np.split(self.train_data, num_train_batches)

        print(" Test  size is: {0}, splitting into {1} batches".format(len(self.test_data), num_test_batches))

        self.test_data = np.split(self.test_data, num_test_batches)
        self.test_targets = te_onehot[:test_cutoff]
        self.test_targets = np.split(self.test_targets, num_test_batches)
        self.test_sequence_lengths = te_sequence_lengths[:test_cutoff]
        self.test_sequence_lengths = np.split(self.test_sequence_lengths, num_test_batches)

        # train and test random indices (the test batches should be random or not?)
        np.random.seed(train_seed)
        self.train_batch_indices = np.arange(0,num_train_batches)
        self.test_batch_indices = np.arange(0,num_test_batches)
        np.random.shuffle(self.train_batch_indices)
        # np.random.shuffle(self.test_batch_indices) # maybe not...

        print("")


    def getBatch(self, test_data=False):
        '''
        Get a random batch of data to preprocess for a step
        not sure how efficient this is...

        Input:
        test_data: flag indicating if a test (True) or a train (False)
                   batch has to be returned

        Returns:
        A numpy arrays for inputs, target, and seq_lengths

        '''
        if not test_data:
            i = self.train_batch_indices[self.train_batch_pointer]
            batch_inputs = self.train_data[i]  # .transpose()
            targets = self.train_targets[i]
            seq_lengths = self.train_sequence_lengths[i]

            # update batch pointer
            self.train_batch_pointer = (self.train_batch_pointer + 1) % len(self.train_data)

            # shuffle batch when all the dataset is traversed
            if self.shuffle_each_pass and self.train_batch_pointer == 0:
                np.random.shuffle(self.train_batch_indices)
            return batch_inputs, targets, seq_lengths
        else:
            i = self.test_batch_indices[self.test_batch_pointer]
            batch_inputs = self.test_data[i]  # .transpose()
            targets = self.test_targets[i]
            seq_lengths = self.test_sequence_lengths[i]

            # update batch pointer
            self.test_batch_pointer = (self.test_batch_pointer + 1) % len(self.test_data)

            # shuffle batch when all the dataset is traversed
            #if self.shuffle_each_pass and self.test_batch_pointer == 0: np.random.shuffle(self.test_batch_indices)
            return batch_inputs, targets, seq_lengths



def test_batch_shuffling(dataH):
    ntb = len(dataH.train_batch_indices)
    for i in range(0,ntb*3):
        if i % ntb == 0:
            print("\nbatch indices now are: {}".format(dataH.train_batch_indices))
            print('batch numbers:         ', end='')
        print(' {}'.format(dataH.train_batch_indices[dataH.train_batch_pointer]), end='')
        batch_inputs, targets, seq_lengths = dataH.getBatch()
        if i == 0: print("\nfirst batch_inputs:\n {}".format(batch_inputs))
    print('\n')


def test():
    np.random.seed(1)
    dataH = DataHandler("../data/processed/", 32,400, True, 3)
    #print("data shape: {}".format(dataH.train_data.shape))
    print("train data shape (batches): list of {} elements with shape {}".format(len(dataH.train_data),dataH.train_data[0].shape))
    print("test  data shape (batches): list of {} elements with shape {}".format(len(dataH.test_data),dataH.test_data[0].shape))
    test_batch_shuffling(dataH)

    bi, targets, seq_lengts = dataH.getBatch()
    print("batch inputs shape: {}, targets shape: {}, seq_lengths shape {}".format(bi.shape,targets.shape,seq_lengts.shape))








if __name__ == '__main__':
    test()
