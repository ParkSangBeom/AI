import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from datetime import datetime
import TensorBoard as tb

class Network:
    _LEARNING_RATE = 0.1
    _EPOCH = 50

    def __init__(self, sess : tf.Session, name : str, data : str):
        self._sess = sess
        self._name = name
        self._tb = tb.TensorBoard(name, sess)

        self._ConvertData(data)
        self._BuildNetwork()
        self._SetTensorBoard()

    def _BuildNetwork(self):    
        self.X = tf.placeholder(tf.int32, [None, self._sequence_length])  # X data
        self.Y = tf.placeholder(tf.int32, [None, self._sequence_length])  # Y label
        x_one_hot = tf.one_hot(self.X, self._num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0

        #cell = tf.contrib.rnn.BasicLSTMCell(num_units = self._hidden_size, state_is_tuple = True)   
        multi_cells = rnn.MultiRNNCell([self.CreateCell() for _ in range(2)], state_is_tuple=True)
        outputs, states = tf.nn.dynamic_rnn(multi_cells, x_one_hot, dtype=tf.float32)

        # FC layer
        X_for_fc = tf.reshape(outputs, [-1, self._hidden_size])
        outputs = tf.contrib.layers.fully_connected(X_for_fc, self._num_classes, activation_fn = None)

        # reshape out for sequence_loss
        outputs = tf.reshape(outputs, [self._batch_size, self._sequence_length, self._num_classes])
        weights = tf.ones([self._batch_size, self._sequence_length])

        sequence_loss = tf.contrib.seq2seq.sequence_loss(logits = outputs, targets = self.Y, weights = weights)
        self._loss = tf.reduce_mean(sequence_loss)
        self._train = tf.train.AdamOptimizer(self._LEARNING_RATE).minimize(self._loss)

        self.prediction = tf.argmax(outputs, axis=2) 

    def CreateCell(self):
        cell = rnn.BasicLSTMCell(num_units = self._hidden_size, state_is_tuple=True)
        return cell

    def _SetTensorBoard(self):
        self._tb.Scalar("Loss_Value", self._loss)
        self._tb.Merge()

    def _ConvertData(self, data):
        self._idx2char = list(set(data))  # index -> char
        self._char2idx = {c : i for i, c in enumerate(self._idx2char)}  # char -> idex

        self._batch_size = 1  # one sample data, one batch
        self._sequence_length = len(data) - 1  # number of lstm rollings (unit #)
        self._dic_size = len(self._char2idx)  # RNN input size (one hot size)

        self._hidden_size = len(self._char2idx)  # RNN output size
        self._num_classes = len(self._char2idx)  # final output size (RNN or softmax, etc.)

        sample_idx = [self._char2idx[c] for c in data]  # char to index
        self._x_data  = [sample_idx[:-1]]  # X data sample (0 ~ n-1) hello: hell
        self._y_data  = [sample_idx[1:]]   # Y label sample (1 ~ n) hello: ello

    def Train(self):
         for i in range(self._EPOCH):
            l, _, summary = self._sess.run([self._loss, self._train, self._tb.merge_summary], feed_dict={self.X: self._x_data, self.Y: self._y_data})
            result = self._sess.run(self.prediction, feed_dict={self.X: self._x_data})
            self._tb.writer.add_summary(summary, global_step = i)

            # print char using dic
            result_str = [self._idx2char[c] for c in np.squeeze(result)]
            print(i, "loss:", l, "Prediction:", ''.join(result_str))