import subprocess
import configparser
import preprocessing.dataprocessor as dp
from util import hyperparams as hyp
import os


def main():
    params = hyp.read_config_file("config.ini")
    data_dir = params["general"]["data_dir"]

    # Generate all the necessary data for training
    dp.process_data(data_dir, params["dataprocessor_params"])
    glove_train_embedding(data_dir, params["glove_params"])


def glove_train_embedding(data_dir, params):
    # Create Co-occurrence Matrix
    subprocess.call("glove/build/cooccur -memory 4.0 -vocab-file " + data_dir + "vocab.txt -verbose "
                    + params["verbose"] + "-symmetric 1 -window-size " + params["window_size"] + " <"
                    + data_dir + "corpus.txt > glove/out/cooccurrence.bin", shell=True)

    # Shuffle Co-occurrence Matrix
    subprocess.call("glove/build/shuffle -memory 4.0 -verbose " + params["verbose"]
                    + " < glove/out/cooccurrence.bin> glove/out/cooccurrence.shuf.bin", shell=True)

    # Run Glove Model
    subprocess.call("glove/build/glove -save-file " + data_dir + "vectors" +
                    " -threads " + params["num_threads"] +
                    " -input-file glove/out/cooccurrence.shuf.bin" +
                    " -iter " + params["max_iter"] +
                    " -vector-size " + params["vector_size"] +
                    " -vocab-file " + data_dir + "vocab.txt" +
                    " -x-max " + params["x_max"] +
                    " -alpha 0.75" +
                    " -eta " + params["learning_rate"] +
                    " -verbose " + params["verbose"] +
                    " -binary " + params["binary"], shell=True)

def convert_gv_to_embedding_matrix(data_dir):
    """converts the embedding matrix and returns the path of the file"""
    # Convert glove vectors into an embedding matrix
    import preprocessing.converter as converter
    converter.embedding_matrix_converter(data_dir + "vectors.txt", data_dir + "embedding_matrix.npy")
    return data_dir + "embedding_matrix.npy"


if __name__ == '__main__':
    main()
