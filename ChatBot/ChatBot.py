import tensorflow as tf
import numpy as np
import Network as nk
import Saver as sv

NAME = "ChatBot"
WORD = "if you want you"

def main(_):
    with tf.Session() as sess:
        network = nk.Network(sess, NAME, WORD)
        sess.run(tf.global_variables_initializer())
        #saver = sv.Saver(NAME, sess)

        network.Train()
if __name__ == "__main__":
    tf.app.run()
