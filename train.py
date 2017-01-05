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
import configparser
from LSTMNet import sentiment
from preprocessing import dataprocessor as dp
from util import hyperparams as hyperparams
import preprocessing.vocabmapping as vmapping

# Defaults for network parameters

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("config_file", "config.ini", "Path to configuration file with hyper-parameters.")
flags.DEFINE_string("checkpoint_dir", "data/checkpoints/", "Directory to store/restore checkpoints")
flags.DEFINE_string("seed", 1, "seed to randomize data")
flags.DEFINE_string("train_seed", 2, "seed to randomize training")



def main():
    hyper_params = read_config_file()
    data_dir = hyper_params["general"]["data_dir"]
    dp_params = hyper_params["dataprocessor_params"]
    processor = dp.DataProcessor(data_dir, dp_params["remove_stopwords"],
                                 dp_params["remove_punct"],
                                 int(dp_params["max_seq_length"]),
                                 int(dp_params["max_vocab_size"]),
                                 int(dp_params["min_vocab_count"]))
    processor.run()

    # create model
    net_params = hyper_params["sentiment_network_params"]
    print("Creating model with...")
    print("Number of hidden layers: {0}".format(net_params["num_layers"]))
    print("Number of units per layer: {0}".format(net_params["hidden_size"]))
    print("Dropout: {0}".format(net_params["dropout"]))

    vocabmapping = vmapping.VocabMapping(data_dir + "vocab.txt")
    vocab_size = vocabmapping.getSize()
    print("Vocab size is: {0}".format(vocab_size))

    path = os.path.join(data_dir, "processed/")

    with tf.Session() as sess:
        # set seeds
        np.random.seed(FLAGS.seed)
        tf.set_random_seed(FLAGS.seed)

        model = create_model(sess, net_params, vocab_size)
        model.initData(path, float(net_params["train_frac"]), -1, True, FLAGS.train_seed)
        writer = tf.train.SummaryWriter("/tmp/tb_logs", sess.graph)


        print("Beggining training...")
        print("Maximum number of epochs to train for: {0}".format(net_params["max_epoch"]))
        print("Batch size: {0}".format(net_params["batch_size"]))
        print("Starting learning rate: {0}".format(net_params["learning_rate"]))
        print("Learning rate decay factor: {0}".format(net_params["lr_decay_factor"]))

        steps_per_checkpoint = int(net_params["steps_per_checkpoint"])
        num_test_batches = len(model.dataH.test_data)
        num_train_batches = len(model.dataH.train_data)
        step_time, loss = 0.0, 0.0
        previous_losses = []
        tot_steps = num_train_batches * int(net_params["max_epoch"])
        # starting at step 1 to prevent test set from running after first batch
        for step in range(1, tot_steps):
            # Get a batch and make a step.
            start_time = time.time()

            inputs, targets, seq_lengths = model.dataH.getBatch()
            str_summary, step_loss, _ = model.step(sess, inputs, targets, seq_lengths)

            step_time += (time.time() - start_time) / steps_per_checkpoint
            loss += step_loss / steps_per_checkpoint

            # Once in a while we print statistics, and run evals.
            if step % steps_per_checkpoint == 0:
                writer.add_summary(str_summary, step)
                # Print statistics for the previous 'epoch'.
                print("global step %d learning rate %.7f step-time %.2f loss %.4f"
                      % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, loss))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                step_time, loss, test_accuracy = 0.0, 0.0, 0.0

                # Run evals on test set and print their accuracy.
                # print("projW:{}".format(model.projW.eval()))
                print("Running test set")
                for test_step in range(num_test_batches):
                    inputs, targets, seq_lengths = model.dataH.getBatch(True)
                    str_summary, test_loss, _, accuracy = model.step(sess, inputs, targets, seq_lengths, train=False)
                    loss += test_loss
                    test_accuracy += accuracy

                norm_test_loss, norm_test_accuracy = loss / num_test_batches, test_accuracy / num_test_batches
                writer.add_summary(str_summary, step)
                print(
                    "Avg Test Loss: {0}, Avg Test Accuracy: {1}".format(norm_test_loss, norm_test_accuracy))
                print("-------Step {0}/{1}------".format(step, tot_steps))
                loss = 0.0 # loss reset
                sys.stdout.flush()


def create_model(session, hyper_params, vocab_size):
    model = sentiment.SentimentModel(vocab_size,
                                     int(hyper_params["hidden_size"]),
                                     int(hyper_params["num_rec_units"]),
                                     float(hyper_params["dropout"]),
                                     int(hyper_params["num_layers"]),
                                     float(hyper_params["grad_clip"]),
                                     int(hyper_params["max_seq_length"]),
                                     float(hyper_params["learning_rate"]),
                                     float(hyper_params["lr_decay_factor"]),
                                     int(hyper_params["batch_size"]))
    session.run(tf.initialize_all_variables())
    return model


def read_config_file():
    config = configparser.ConfigParser()
    config.read(FLAGS.config_file)
    return config


if __name__ == '__main__':
    main()
