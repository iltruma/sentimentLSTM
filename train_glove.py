import subprocess
import configparser

def main():
    params = read_config_file()
    data_dir = params["general"]["data_dir"]


    #Create Co-occurrence Matrix
    subprocess.call("glove/build/cooccur -memory 4.0 -vocab-file " + data_dir + "vocab.txt -verbose " + params["glove_params"]["verbose"] + "-symmetric 1 -window-size " + params["glove_params"]["window_size"] + " <" + data_dir + "corpus.txt > glove/out/cooccurrence.bin", shell=True)

    #Shuffle Co-occurrence Matrix
    subprocess.call("glove/build/shuffle -memory 4.0 -verbose " + params["glove_params"]["verbose"] + " < glove/out/cooccurrence.bin> glove/out/cooccurrence.shuf.bin", shell=True)

    #Run Glove Model
    subprocess.call("glove/build/glove -save-file " + data_dir + "vectors" +
                                     " -threads " + params["glove_params"]["num_threads"] +
                                     " -input-file glove/out/cooccurrence.shuf.bin" +
                                     " -iter " + params["glove_params"]["max_iter"] +
                                     " -vector-size " + params["glove_params"]["vector_size"] +
                                     " -vocab-file " + data_dir + "vocab.txt" +
                                     " -x-max " + params["glove_params"]["x_max"] +
                                     " -alpha 0.75" +
                                     " -eta " + params["glove_params"]["learning_rate"] +
                                     " -verbose " + params["glove_params"]["verbose"] +
                                     " -binary " + params["glove_params"]["binary"], shell=True)

    #Convert glove vectors into an embedding matrix
    import preprocessing.converter as converter
    converter.embedding_matrix_converter(data_dir + "vectors.txt", data_dir + "embedding_matrix.npy")


def read_config_file():
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config


if __name__ == '__main__':
    main()
