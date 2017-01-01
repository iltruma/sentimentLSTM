import os
import numpy as np


class DataHandler(object):
    def __init__(self, data_path, batch_size, train_frac, max_examples=-1, shuffle_each_pass=True, train_seed=None):
        print("init DataHandler with path:{}, batch_size:{}, training fraction ".format(data_path,batch_size)
              + "{}, max number of examples:{}".format(train_frac,max_examples))
        self.batch_size = batch_size
        self.shuffle_each_pass = shuffle_each_pass


        # input files (numpy matrices, 1 row per example)
        inFiles = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

        # load all files in memory
        self.data = np.load(os.path.join(data_path, inFiles[0]))
        for i in range(1, len(inFiles)):
            self.data = np.vstack((self.data, np.load(os.path.join(data_path, inFiles[i]))))

        # randomize the dataset
        np.random.shuffle(self.data)

        # use only max_example examples when defined
        if max_examples >-1 and max_examples < len(self.data):
            self.data = self.data[:max_examples]

        self.num_batches = len(self.data) // self.batch_size

        # 70/30 split for train/test
        train_start_end_index = [0, int(train_frac * len(self.data))]
        test_start_end_index = [int(train_frac * len(self.data)) + 1, len(self.data) - 1]

        self.train_batch_pointer = 0
        self.test_batch_pointer = 0

        targets = (self.data.transpose()[-2]).transpose()
        onehot = np.zeros((len(targets), 2))
        onehot[np.arange(len(targets)), targets] = 1
        sequence_lengths = (self.data.transpose()[-1]).transpose()
        self.data = (self.data.transpose()[0:-2]).transpose()


        self.train_data = self.data[train_start_end_index[0]: train_start_end_index[1]]
        self.test_data = self.data[test_start_end_index[0]:test_start_end_index[1]]

        # cutoff non even number of batches (is it necessary?)
        num_train_batches = len(self.train_data) // self.batch_size
        num_test_batches = len(self.test_data) // self.batch_size
        train_cutoff = len(self.train_data) - (len(self.train_data) % self.batch_size)
        test_cutoff = len(self.test_data) - (len(self.test_data) % self.batch_size)
        self.train_data = self.train_data[:train_cutoff]
        self.test_data = self.test_data[:test_cutoff]

        print(" Train size is: {0}, splitting into {1} batches".format(len(self.train_data), num_train_batches))

        self.train_sequence_lengths = sequence_lengths[train_start_end_index[0]:train_start_end_index[1]][:train_cutoff]
        self.train_sequence_lengths = np.split(self.train_sequence_lengths, num_train_batches)
        self.train_targets = onehot[train_start_end_index[0]:train_start_end_index[1]][:train_cutoff]
        self.train_targets = np.split(self.train_targets, num_train_batches)
        self.train_data = np.split(self.train_data, num_train_batches)

        print(" Test  size is: {0}, splitting into {1} batches".format(len(self.test_data), num_test_batches))

        self.test_data = np.split(self.test_data, num_test_batches)
        self.test_targets = onehot[test_start_end_index[0]:test_start_end_index[1]][:test_cutoff]
        self.test_targets = np.split(self.test_targets, num_test_batches)
        self.test_sequence_lengths = sequence_lengths[test_start_end_index[0]:test_start_end_index[1]][:test_cutoff]
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
        dataH.getBatch()
    print('\n')


def test():
    dataH = DataHandler("../data/processed/", 16, 0.7,400, True)
    print("data shape: {}".format(dataH.data.shape))
    print("train data shape (batches): list of {} elements with shape {}".format(len(dataH.train_data),
                                                                                 dataH.train_data[0].shape))
    print("test  data shape (batches): list of {} elements with shape {}".format(len(dataH.test_data),
                                                                                 dataH.test_data[0].shape))
    test_batch_shuffling(dataH)

    bi, targets, seq_lengts = dataH.getBatch()
    print("batch inputs shape: {}, targets shape: {}, seq_lengths shape {}".format(bi.shape,targets.shape,
                                                                                       seq_lengts.shape))










if __name__ == '__main__':
    test()
