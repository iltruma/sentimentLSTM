"""
I used mainly the tensorflow translation example:
https://github.com/tensorflow/tensorflow/

and semi-based this off the sentiment analyzer here:
http://deeplearning.net/tutorial/lstm.html
"""

import tensorflow as tf
import numpy as np
import sys
import os
import time
from LSTMNet import sentiment
from preprocessing import dataprocessor as dp
from util import hyperparams as hyp
import preprocessing.vocabmapping as vmapping


# Defaults for network parameters

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("config_file", "config.ini", "Path to configuration file with hyper-parameters.")
flags.DEFINE_string("checkpoint_dir", "data/checkpoints/", "Directory to store/restore checkpoints")
flags.DEFINE_string("seed", None, "seed to randomize data")
flags.DEFINE_string("train_seed", None, "seed to randomize training")


def test_net():
    print("----Test NN with random embedding----")
    hyper_params = hyp.read_config_file(FLAGS.config_file)
    data_dir = hyper_params["general"]["data_dir"]
    dp_params = hyper_params["dataprocessor_params"]

    # generate all the necessary data for training
    dp.process_data(data_dir, dp_params)

    # trains the neural net with the config.ini hyperparameters and the data already generated
    train_nn(data_dir, hyper_params["sentiment_network_params"])


def test_net_with_glove():
    print("----Test NN with GLOVE embedding----")
    import train_glove as glove

    params = hyp.read_config_file("config.ini")
    if params["glove_params"]["vector_size"] != params["sentiment_network_params"]["embedding_dim"]:
        print("ERROR: glove vector size and nn embedding size are different! change them to be equal in config.ini")
        return

    data_dir = params["general"]["data_dir"]
    embedding_matrix_path = data_dir + "embedding_matrix.npy"

    # Generate all the necessary data for training
    processor = dp.process_data(data_dir, params["dataprocessor_params"])
    changed_signature = processor.changed_signature

    if changed_signature or not os.path.exists(embedding_matrix_path):
        # Train glove embedding matrix and convert it
        glove.train_and_convert(data_dir, params["glove_params"])
    else:
        print("embedding_matrix already found...")

    embedding_matrix = np.load(embedding_matrix_path)
    if len(embedding_matrix.transpose()) != int(params["glove_params"]["vector_size"]):
        print("matrix must be made again due to inconsistent embedding dimension")
        # Train glove embedding matrix and convert it
        glove.train_and_convert(data_dir, params["glove_params"])
        embedding_matrix = np.load(embedding_matrix_path)

    # Train the Neural net with the embedding matrix given by glove
    train_nn(data_dir, params["sentiment_network_params"], embedding_matrix)


def train_nn(data_dir, net_params, embedding_matrix=None, train_embedding=False):
    # create model
    print("Creating model with...")
    hyp.print_parameters(net_params) # print networks parameter

    vocabmapping = vmapping.VocabMapping(data_dir + "vocab.txt")
    vocab_size = vocabmapping.getSize()
    print("Vocab size: {}".format(vocab_size))

    path = os.path.join(data_dir, "processed/")

    with tf.Session() as sess:
        # set seeds
        np.random.seed(FLAGS.seed)
        tf.set_random_seed(FLAGS.seed)

        model = create_model(sess, net_params, vocab_size, embedding_matrix)
        model.initData(path, -1, True, FLAGS.train_seed)

        lt = time.localtime()
        logName = "tlog" + str(lt.tm_hour) + ":" + str(lt.tm_min) + ":" + str(lt.tm_sec) +  \
                  "_" + str(lt.tm_mday) + ":" + str(lt.tm_mon) + ":" + str(lt.tm_year)

        os.makedirs("tblogs/" + logName)
        train_writer = tf.summary.FileWriter("tblogs/" + logName + "/train", sess.graph)
        test_writer = tf.summary.FileWriter("tblogs/" + logName + "/test", sess.graph)

        print("Beginning training...")

        # summary graph... it's ugly but works :/
        with tf.variable_scope("summary"), tf.device("/cpu:0"):
            loss_summ_var = tf.placeholder(tf.float32, name="loss_summ_var")
            acc_summ_var = tf.placeholder(tf.float32, name="acc_summ_var")
            loss_summ = tf.summary.scalar("loss", loss_summ_var)
            acc_summ = tf.summary.scalar("accuracy", acc_summ_var)
            summaries = tf.summary.merge([loss_summ, acc_summ])

        steps_per_checkpoint = int(net_params["steps_per_checkpoint"])
        num_test_batches = len(model.dataH.test_data)
        num_train_batches = len(model.dataH.train_data)
        step_time, loss, train_accuracy = 0.0, 0.0, 0.0


        previous_losses = []
        max_epoch = int(net_params["max_epoch"])
        tot_steps = num_train_batches * max_epoch
        # starting at step 1 to prevent test set from running after first batch
        for step in range(1, tot_steps):
            # Get a batch and make a step.
            start_time = time.time()

            inputs, targets, seq_lengths = model.dataH.getBatch()
            str_summary, step_loss, _, step_accuracy = model.step(sess, inputs, targets, seq_lengths)

            step_time += (time.time() - start_time) / steps_per_checkpoint
            loss += step_loss / steps_per_checkpoint
            train_accuracy += step_accuracy / steps_per_checkpoint

            # Once in a while we print statistics, and run evals.
            if step % steps_per_checkpoint == 0:
                n_epoch = (step // num_train_batches) + 1

                # writing summary using summary graph created above, it's ugly but works :/
                train_writer.add_summary(sess.run(summaries, {loss_summ_var.name: loss, acc_summ_var:train_accuracy}), step)

                # Print statistics for the previous 'checkpoint-epoch'.
                print("global step %d learning rate %.7f step-time %.2f loss %.4f, accuracy: %4f"
                      % (step, model.learning_rate.eval(),
                         step_time, loss, train_accuracy))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # zero timer loss and accuracy
                step_time, loss, train_accuracy, test_accuracy = 0.0, 0.0, 0.0, 0.0

                # Run evals on test set and print their accuracy.
                print("Running test set")
                for test_step in range(num_test_batches):
                    inputs, targets, seq_lengths = model.dataH.getBatch(True)
                    str_summary, test_loss, _, accuracy = model.step(sess, inputs, targets, seq_lengths, train=False)
                    loss += test_loss
                    test_accuracy += accuracy

                norm_test_loss, norm_test_accuracy = loss / num_test_batches, test_accuracy / num_test_batches

                # writing summary using summary graph created above, it's ugly but works :/
                test_writer.add_summary(sess.run(summaries, {loss_summ_var.name: norm_test_loss,
                                                             acc_summ_var: norm_test_accuracy}), step)
                print(
                    "Avg Test Loss: {}, Avg Test Accuracy: {}".format(norm_test_loss, norm_test_accuracy))
                print("-------Step {}/{}--epoch:{}/{}".format(step, tot_steps, n_epoch, max_epoch))
                loss = 0.0  # loss reset
                sys.stdout.flush()


def create_model(session, hyper_params, vocab_size, embedding_matrix=None):
    model = sentiment.SentimentModel(vocab_size,
                                     int(hyper_params["embedding_dim"]),
                                     int(hyper_params["num_rec_units"]),
                                     int(hyper_params["hidden_dim"]),
                                     float(hyper_params["dropout"]),
                                     int(hyper_params["num_rec_layers"]),
                                     float(hyper_params["grad_clip"]),
                                     int(hyper_params["max_seq_length"]),
                                     float(hyper_params["learning_rate"]),
                                     float(hyper_params["lr_decay_factor"]),
                                     int(hyper_params["batch_size"]),
                                     embedding_matrix=embedding_matrix,
                                     train_embedding=hyper_params["train_embedding"] == 'True')
    session.run(tf.global_variables_initializer())
    return model


if __name__ == '__main__':
    test_net_with_glove()
    # test_net()
