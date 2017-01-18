import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell, seq2seq
import numpy as np
from LSTMNet import datahandler as dh


class SentimentModel(object):
    def __init__(self, vocab_size, embedding_dim, num_rec_units, hidden_dim, dropout,
                 num_rec_layers, max_gradient_norm, max_seq_length,
                 learning_rate, lr_decay, batch_size, forward_only=False, embedding_matrix=None, train_embedding=False):
        self.num_classes = 2
        self.vocab_size = vocab_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * lr_decay)
        self.batch_pointer = 0
        self.seq_input = []
        self.batch_size = batch_size
        self.seq_lengths = []
        self.embedding_dim = embedding_dim
        self.num_rec_units = num_rec_units
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.max_gradient_norm = max_gradient_norm
        self.global_step = tf.Variable(0, trainable=False)
        self.max_seq_length = max_seq_length


        # seq_input: list of tensors, each tensor is size max_seq_length
        # target: a list of values betweeen 0 and 1 indicating target scores
        # seq_lengths:the early stop lengths of each input tensor
        self.seq_input = tf.placeholder(tf.int32, shape=[None, max_seq_length], name="input")
        self.target = tf.placeholder(tf.float32, name="target", shape=[None, self.num_classes])
        self.seq_lengths = tf.placeholder(tf.int32, shape=[None], name="early_stop")

        self.dropout_keep_prob_embedding = tf.constant(self.dropout)
        self.dropout_keep_prob_lstm_input = tf.constant(self.dropout)
        self.dropout_keep_prob_lstm_output = tf.constant(self.dropout)

        # embedding weights
        embedded_tokens_drop = self.embedding_layer(embedding_matrix, train_embedding)

        lstm_input = [embedded_tokens_drop[:, i, :] for i in range(self.max_seq_length)]

        self.rnn_output, self.rnn_state = self.lstm_layers_average(lstm_input, num_rec_layers)


        # hidden layer as in the adversarial paper
        if self.hidden_dim > 0:
            with tf.variable_scope("hidden_layer"):
                W = tf.get_variable(
                    "W",
                    [self.num_rec_units, self.hidden_dim],
                    initializer=tf.truncated_normal_initializer(stddev=0.1))
                b = tf.get_variable(
                    "b",
                    [self.hidden_dim],
                    initializer=tf.constant_initializer(0.1))
                self.hidden_output = tf.nn.relu(tf.nn.xw_plus_b(self.rnn_output, W, b))
        else:
            self.hidden_dim =self.num_rec_units
            self.hidden_output = self.rnn_output


        with tf.variable_scope("output_projection"):
            W = tf.get_variable(
                "W",
                [self.hidden_dim, self.num_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable(
                "b",
                [self.num_classes],
                initializer=tf.constant_initializer(0.1))
            self.scores = tf.nn.xw_plus_b(self.hidden_output, W, b)
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
                opt = tf.train.RMSPropOptimizer(self.learning_rate)
                #opt = tf.train.AdamOptimizer(self.learning_rate)
            gradients = tf.gradients(self.losses, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            with tf.name_scope("grad_norms"):
                grad_summ = tf.summary.scalar("grad_norms", norm)
            self.update = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
            loss_summ = tf.summary.scalar("loss", self.mean_loss)
            acc_summ = tf.summary.scalar("accuracy", self.accuracy)
            self.merged = tf.summary.merge([loss_summ, acc_summ])
        self.saver = tf.train.Saver(tf.global_variables())

    def initData(self, data_path, max_examples=-1, shuffle_each_pass=True, train_seed=None):
        self.dataH = dh.DataHandler(data_path, self.batch_size, max_examples, shuffle_each_pass, train_seed)

    def embedding_layer(self, pre_W, train_embedding=True):
        with tf.variable_scope("embedding"):
            if pre_W is not None:
                W = tf.Variable(pre_W, trainable=train_embedding, dtype=tf.float32)
            else:
                W = tf.get_variable(
                    "W",
                    [self.vocab_size, self.embedding_dim],
                    initializer=tf.truncated_normal_initializer(stddev=0.1))
            embedded_tokens = tf.nn.embedding_lookup(W, self.seq_input)
            return  tf.nn.dropout(embedded_tokens, self.dropout_keep_prob_embedding)



    def lstm_layers(self, lstm_input, num_rec_layers):
        with tf.variable_scope("lstm"):
            single_cell = rnn_cell.DropoutWrapper(
                rnn_cell.LSTMCell(self.num_rec_units,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1),
                                  state_is_tuple=True),
                input_keep_prob=self.dropout_keep_prob_lstm_input,
                output_keep_prob=self.dropout_keep_prob_lstm_output)
            cell = rnn_cell.MultiRNNCell([single_cell] * num_rec_layers, state_is_tuple=True)

            initial_state = cell.zero_state(self.batch_size, tf.float32)

            outputs, state = rnn.rnn(cell, lstm_input,
                                     initial_state=initial_state,
                                     sequence_length=self.seq_lengths)


            return state[-1][1], state

    def lstm_layers_average(self, lstm_input, num_rec_layers):
        with tf.variable_scope("lstm"):
            print("using LSTM Cell")
            single_cell = rnn_cell.DropoutWrapper(
                rnn_cell.LSTMCell(self.num_rec_units, initializer=tf.truncated_normal_initializer(stddev=0.1)),
                # rnn_cell.GRUCell(self.num_rec_units),
                input_keep_prob=self.dropout_keep_prob_lstm_input,
                output_keep_prob=self.dropout_keep_prob_lstm_output)
            cell = rnn_cell.MultiRNNCell([single_cell] * num_rec_layers, state_is_tuple=True)

            initial_state = cell.zero_state(self.batch_size, tf.float32)

            outputs, state =  rnn.rnn(cell, lstm_input,
                           initial_state=initial_state,
                           sequence_length=self.seq_lengths)
            # right average
            avg_output = tf.reduce_sum(outputs, 0)
            for i in range(self.batch_size):
                tf.truediv(avg_output[i, :],  tf.cast(self.seq_lengths[i], tf.float32))

            # wrong average
            avg_output = tf.reduce_mean(outputs, 0)

            return avg_output, state

    def lstm_layers_weighted_average(self, lstm_input, num_rec_layers):
        with tf.variable_scope("lstm"):
            single_cell = rnn_cell.DropoutWrapper(
                rnn_cell.LSTMCell(self.num_rec_units,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1),
                                  state_is_tuple=True),
                input_keep_prob=self.dropout_keep_prob_lstm_input,
                output_keep_prob=self.dropout_keep_prob_lstm_output)
            cell = rnn_cell.MultiRNNCell([single_cell] * num_rec_layers, state_is_tuple=True)

            initial_state = cell.zero_state(self.batch_size, tf.float32)

            outputs, state = rnn.rnn(cell, lstm_input,
                                     initial_state=initial_state,
                                     sequence_length=self.seq_lengths)

            avg_output = tf.reduce_sum(outputs, 0)
            for i in range(self.batch_size):
                c = 1/tf.constant(tf.reduce_sum(tf.range(self.seq_lengths[i]+1)), dtype=tf.float32)
                weights = tf.mul(tf.range(1, self.max_seq_length+1), c)
                tf.sub()



                tf.truediv(avg_output[i, :], tf.cast(self.seq_lengths[i], tf.float32))

            return avg_output, state


    def nn_layer(self, input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        """Reusable code for making a simple neural net layer.

        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        """
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = tf.get_variable(
                    "W",
                    [input_dim, output_dim],
                    initializer=tf.random_uniform_initializer(-1.0, 1.0))
            with tf.name_scope('biases'):
                biases = tf.get_variable(
                    "b",
                    [output_dim],
                    initializer=tf.constant_initializer(0.1))
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.nn.xw_plus_b(input_tensor, weights, biases)
            activations = act(preactivate, name='activation')
            return activations

    def step(self, session, inputs, targets, seq_lengths, train=True):
        '''
        Inputs:
        session: tensorflow session
        inputs: list of list of ints representing tokens in review of batch_size
        output: list of sentiment scores
        seq_lengths: list of sequence lengths, provided at runtime to prevent need for padding

        Returns:
        merged_tb_vars, loss, none
        or (in train=False):
        merged_tb_vars, loss, outputs, accuracy
        '''
        input_feed = {}
        input_feed[self.seq_input.name] = inputs
        input_feed[self.target.name] = targets
        input_feed[self.seq_lengths.name] = seq_lengths
        if train:
            output_feed = [self.merged, self.mean_loss, self.update, self.accuracy]
            outputs = session.run(output_feed, input_feed)
            return outputs[0], outputs[1], None, outputs[3]

        else:
            output_feed = [self.merged, self.mean_loss, self.y, self.accuracy]
            outputs = session.run(output_feed, input_feed)
            return outputs[0], outputs[1], outputs[2], outputs[3]


def test():
    test_steps = 10
    dataDir = "../data/"
    seed = 1  # data will always be randomized in the same way for the training/test division
    train_seed = 3  # given the above division the training will chose from the sets differently changing this
    print("sentiment.py test with test_steps: {}, seed: {}, train_seed: {}".format(test_steps, seed, train_seed))

    from preprocessing.vocabmapping import VocabMapping
    vocab_size = VocabMapping(dataDir + "vocab.txt").getSize()
    print("vocabulary size is {}".format(vocab_size))

    sess = tf.Session()
    np.random.seed(seed)
    tf.set_random_seed(seed)
    print("tensorflow session started + tf and numpy seed set")
    model = SentimentModel(vocab_size=vocab_size, embedding_dim=50, num_rec_units=100, hidden_dim=30, dropout=0.5,
                           num_rec_layers=2, max_gradient_norm=5, max_seq_length=200,
                           learning_rate=0.01, lr_decay=0.97, batch_size=16, forward_only=False,
                           embedding_matrix=np.random.rand(vocab_size, 50))
    print("Created model")

    model.initData(dataDir + "processed/", 0.7, 400, True, train_seed)
    print("dataset initialized")

    sess.run(tf.initialize_all_variables())
    print("varables initialized")

    for i in range(0, test_steps):
        inputs, targets, seq_lengths = model.dataH.getBatch()
        str_summary, step_loss, _ = model.step(sess, inputs, targets, seq_lengths)
        print("{}th step executed! results:".format(i))
        print(" step_loss: %.4f" % step_loss)



if __name__ == '__main__':
    test()
