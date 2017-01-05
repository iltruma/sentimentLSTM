import subprocess
import configparser
import preprocessing.dataprocessor as dp
from util import hyperparams as hyp
import os
import numpy as np
import train_glove as glove
import train as train_net

def main():
    params = hyp.read_config_file("config.ini")
    data_dir = params["general"]["data_dir"]

    # Generate all the necessary data for training
    dp.process_data(data_dir, params["dataprocessor_params"])

    # Train glove embedding matrix
    glove.glove_train_embedding(data_dir, params["glove_params"])

    # Convert the embedding matrix to be used with
    embedding_matrix_path = glove.convert_gv_to_embedding_matrix(data_dir)
    embedding_matrix = np.load(embedding_matrix_path)

    # Train the Neural net with the embedding matrix given by glove
    train_net.train_nn(data_dir, params["sentiment_network_params"], embedding_matrix, train_embedding=False)


if __name__ == '__main__':
    main()

