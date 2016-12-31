import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell, seq2seq
import numpy as np


class SentimentModel(object):
    def __init__(self, vocab_size, hidden_size, dropout,
                 num_layers, max_gradient_norm, max_seq_length,
                 learning_rate, lr_decay, batch_size, forward_only=False):
        self.num_classes = 2
        self.vocab_size = vocab_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * lr_decay)
        self.batch_pointer = 0
        self.seq_input = []
        self.batch_size = batch_size
        self.seq_lengths = []
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.max_gradient_norm = max_gradient_norm
        self.global_step = tf.Variable(0, trainable=False)
        self.max_seq_length = max_seq_length

        self.str_summary_type = tf.placeholder(tf.string, name="str_summary_type")

        # seq_input: list of tensors, each tensor is size max_seq_length
        # target: a list of values betweeen 0 and 1 indicating target scores
        # seq_lengths:the early stop lengths of each input tensor
        self.seq_input = tf.placeholder(tf.int32, shape=[None, max_seq_length], name="input")
        self.target = tf.placeholder(tf.float32, name="target", shape=[None, self.num_classes])
        self.seq_lengths = tf.placeholder(tf.int32, shape=[None], name="early_stop")

        self.dropout_keep_prob_embedding = tf.constant(self.dropout)
        self.dropout_keep_prob_lstm_input = tf.constant(self.dropout)
        self.dropout_keep_prob_lstm_output = tf.constant(self.dropout)

        #embedding weights
        embedded_tokens_drop = self.embedding_layer()

    def embedding_layer(self):
        with tf.variable_scope("embedding"), tf.device("/cpu:0"):
            W = tf.get_variable(
                "W",
                [self.vocab_size, self.hidden_size],
                initializer=tf.random_uniform_initializer(-1.0, 1.0))
            embedded_tokens = tf.nn.embedding_lookup(W, self.seq_input)
            return tf.nn.dropout(embedded_tokens, self.dropout_keep_prob_embedding)