import tensorflow as tf
from datetime import datetime

class TensorBoard:

    _tensorboard_path = "./Tensorboard"

    def __init__(self, name : str, sess : tf.Session):
        self._sess = sess
        self._name = name;

        self._WriterSetting()

    def _WriterSetting(self):
        now = datetime.now()
        self._dir = self._tensorboard_path + "/" + self._name + "/" + \
        "(" + str(now.year) + "_" + str(now.month) + "_" + str(now.day) + ")" + str(now.hour) + "_" + str(now.minute) + "_" + str(now.second)

    def Scalar(self, name : str, value):
         tf.summary.scalar(name, value)

    def Histogram(self, name : str, value):
         tf.summary.histogram(name, value)

    def Merge(self):
        self.merge_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self._dir, self._sess.graph)