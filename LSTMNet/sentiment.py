import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell, seq2seq
import numpy as np
import datahandler as dh


class SentimentModel(object):
    def __init__(self, vocab_size, hidden_size, dropout,
                 num_layers, max_gradient_norm, max_seq_length,
                 learning_rate, lr_decay, batch_size, forward_only=False, embedding_matrix=None):
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
        embedded_tokens_drop = self.embedding_layer(embedding_matrix)

        lstm_input = [embedded_tokens_drop[:, i, :] for i in range(self.max_seq_length)]

        rnn_output, rnn_state = self.lstm_layers(lstm_input,num_layers)


        with tf.variable_scope("output_projection"):
            W = tf.get_variable(
                "W",
                [hidden_size, self.num_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable(
                "b",
                [self.num_classes],
                initializer=tf.constant_initializer(0.1))
            self.scores = tf.nn.xw_plus_b(rnn_state[-1][0], W, b)
            self.y = tf.nn.softmax(self.scores)
            self.predictions = tf.argmax(self.scores, 1)

        with tf.variable_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.target, name="ce_losses")
            self.total_loss = tf.reduce_sum(self.losses)
            self.mean_loss = tf.reduce_mean(self.losses)

        with tf.variable_scope("accuracy"):
            self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.target, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

        params = tf.trainable_variables()
        if not forward_only:
            with tf.name_scope("train"):
                opt = tf.train.AdamOptimizer(self.learning_rate)
            gradients = tf.gradients(self.losses, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            with tf.name_scope("grad_norms"):
                grad_summ = tf.summary.scalar("grad_norms", norm)
            self.update = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
            loss_summ = tf.summary.scalar("{0}_loss".format(self.str_summary_type), self.mean_loss)
            acc_summ = tf.summary.scalar("{0}_accuracy".format(self.str_summary_type), self.accuracy)
            self.merged = tf.summary.merge([loss_summ, acc_summ])
        self.saver = tf.train.Saver(tf.global_variables())

    def initData(self, data_path, train_frac, max_examples=-1, shuffle_each_pass = True, train_seed=None):
        self.dataH = dh.DataHandler(data_path, self.batch_size, train_frac, max_examples, shuffle_each_pass, train_seed)

    def embedding_layer(self, pre_W):
        with tf.variable_scope("embedding"), tf.device("/cpu:0"):
            if pre_W is not None:
                W = tf.Variable(pre_W, dtype=tf.float32)
            else:
                W = tf.get_variable(
                    "W",
                    [self.vocab_size, self.hidden_size],
                    initializer=tf.random_uniform_initializer(-1.0, 1.0))
            embedded_tokens = tf.nn.embedding_lookup(W, self.seq_input)
            return tf.nn.dropout(embedded_tokens, self.dropout_keep_prob_embedding)

    def lstm_layers(self, lstm_input, num_layers):
        with tf.variable_scope("lstm"):
            single_cell = rnn_cell.DropoutWrapper(
                rnn_cell.LSTMCell(self.hidden_size,
                                  initializer=tf.random_uniform_initializer(-1.0, 1.0),
                                  state_is_tuple=True),
                input_keep_prob=self.dropout_keep_prob_lstm_input,
                output_keep_prob=self.dropout_keep_prob_lstm_output)
            cell = rnn_cell.MultiRNNCell([single_cell] * num_layers, state_is_tuple=True)

            initial_state = cell.zero_state(self.batch_size, tf.float32)

            return rnn.rnn(cell, lstm_input,
                                            initial_state=initial_state,
                                            sequence_length=self.seq_lengths)


    def step(self, session, inputs, targets, seq_lengths, train=True):
        '''
        Inputs:
        session: tensorflow session
        inputs: list of list of ints representing tokens in review of batch_size
        output: list of sentiment scores
        seq_lengths: list of sequence lengths, provided at runtime to prevent need for padding

        Returns:
        merged_tb_vars, loss, none
        or (in forward only):
        merged_tb_vars, loss, outputs
        '''
        input_feed = {}
        input_feed[self.seq_input.name] = inputs
        input_feed[self.target.name] = targets
        input_feed[self.seq_lengths.name] = seq_lengths
        if train:
            input_feed[self.str_summary_type.name] = "train"
            output_feed = [self.merged, self.mean_loss, self.update]
        else:
            input_feed[self.str_summary_type.name] = "test"
            output_feed = [self.merged, self.mean_loss, self.y, self.accuracy]
        outputs = session.run(output_feed, input_feed)
        if train:
            return outputs[0], outputs[1], None
        else:
            return outputs[0], outputs[1], outputs[2], outputs[3]


def test():
    test_steps = 10
    dataDir = "../data/"
    seed = 1          # data will always be randomized in the same way for the training/test division
    train_seed = 3    # given the above division the training will chose from the sets differently changing this
    print("sentiment.py test with test_steps: {}, seed: {}, train_seed: {}".format(test_steps, seed, train_seed))

    from preprocessing.vocabmapping import VocabMapping
    vocab_size = VocabMapping(dataDir + "vocab.txt").getSize() -1 # -1 for <PAD>
    print("vocabulary size is {}".format(vocab_size))

    sess = tf.Session()
    np.random.seed(seed)
    tf.set_random_seed(seed)
    print("tensorflow session started + tf and numpy seed set")
    model = SentimentModel( vocab_size=vocab_size, hidden_size=50, dropout=0.5,
                 num_layers=1, max_gradient_norm=5, max_seq_length=200,
                 learning_rate=0.01, lr_decay=0.97, batch_size=16, forward_only=False,
                            embedding_matrix=np.random.rand(vocab_size, 50))
    print("Created model")

    model.initData("../data/processed/", 0.7,400, True, train_seed)
    print("dataset initialized")

    sess.run(tf.global_variables_initializer())
    print("varables initialized")

    for i in range(0,test_steps):
        inputs, targets, seq_lengths = model.dataH.getBatch()
        str_summary, step_loss, _ = model.step(sess, inputs, targets, seq_lengths)
        print("{}th step executed! results:".format(i))
        print(" step_loss: %.4f" % step_loss)





if __name__ == '__main__':
    test()


