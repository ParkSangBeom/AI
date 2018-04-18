import tensorflow as tf
import numpy as np
import Network as nk
import Saver as sv
import DataConverter as dc

NAME = "ChatBot"
WORD = "if you want you"

DATA_PATH = "./Data/chat.log"

def main(_):

    data_converter = dc.DataConverter(DATA_PATH)

    with tf.Session() as sess:
        network = nk.Network(sess, NAME, data_converter)
        sess.run(tf.global_variables_initializer())
        #saver = sv.Saver(NAME, sess)

        network.Train(data_converter)

if __name__ == "__main__":
    tf.app.run()
