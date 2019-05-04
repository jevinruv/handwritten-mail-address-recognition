import tensorflow as tf
import os
import shutil

from Constants import Constants


class Model:

    def __init__(self, char_list):
        "initialize model, CNN, RNN, CTC and TensorFlow"

        self.path_model = Constants.path_model
        self.batch_size = Constants.batch_size
        # self.char_list = Constants.file_char_list
        self.char_list = char_list
        self.learning_rate = Constants.learning_rate
        self.text_length = Constants.text_length
        self.img_size = Constants.img_size
        self.snap_id = 0

        self.input_imgs = tf.placeholder(tf.float32, shape=(self.batch_size, self.img_size[0], self.img_size[1]))

        # CNN
        cnn_output = self.build_CNN(self.input_imgs)

        # RNN
        rnn_output = self.build_RNN(cnn_output)

        # CTC
        (self.loss, self.decoder) = self.build_CTC(rnn_output)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # initialize TensorFlow
        (self.sess, self.saver) = self.init_TensorFlow()

    def build_CNN(self, input_cnn):

        input_4d = tf.expand_dims(input=input_cnn, axis=3)  # adds dimensions of size 1, here it adds to the 3rd index

        pool = input_4d

        pool = self.create_CNN_layer(pool, filter_size=5, in_features=1, out_features=32, max_pool=(2, 2))
        pool = self.create_CNN_layer(pool, filter_size=5, in_features=32, out_features=64, max_pool=(2, 2))
        pool = self.create_CNN_layer(pool, filter_size=3, in_features=64, out_features=128, max_pool=(1, 2))
        pool = self.create_CNN_layer(pool, filter_size=3, in_features=128, out_features=128, max_pool=(1, 2))
        pool = self.create_CNN_layer(pool, filter_size=3, in_features=128, out_features=256, max_pool=(1, 2))

        return pool

    def create_CNN_layer(self, pool, filter_size, in_features, out_features, max_pool):

        # initialize weights
        filter = tf.Variable(tf.truncated_normal([filter_size, filter_size, in_features, out_features], stddev=0.1))

        conv = tf.nn.conv2d(pool, filter, padding='SAME', strides=(1, 1, 1, 1))
        relu = tf.nn.relu(conv)
        pool = tf.nn.max_pool(relu,
                              ksize=(1, max_pool[0], max_pool[1], 1),
                              strides=(1, max_pool[0], max_pool[1], 1),
                              padding='VALID')
        # layer 1
        # filter = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
        # conv = tf.nn.conv2d(input=pool, filter=filter, padding='SAME', strides=(1, 1, 1, 1)) # strides=[1, 1, 1, 1], the filter window will move 1 batch, 1 height pixel, 1 width pixel and 1 color pixel
        # relu = tf.nn.relu(conv)
        # pool = tf.nn.max_pool(relu, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID')

        return pool

    def build_RNN(self, input_rnn):

        input_3d = tf.squeeze(input_rnn, axis=[2])  # removes dimensions of size 1, here it removes the 2nd index

        n_hidden = 256
        n_layers = 2
        cells = []

        for _ in range(n_layers):
            cells.append(tf.nn.rnn_cell.LSTMCell(num_units=n_hidden))

        cell_stack = tf.nn.rnn_cell.MultiRNNCell(cells)  # combine the 2 LSTMCell created

        # BxTxF -> BxTx2H
        ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_stack,
                                                        cell_bw=cell_stack,
                                                        inputs=input_3d,
                                                        dtype=input_3d.dtype)

        rnn = tf.concat([fw, bw], 2)  # BxTxH + BxTxH -> BxTx2H
        concat = tf.expand_dims(rnn, 2)  # BxTx2H -> BxTx1X2H

        # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        kernel = tf.Variable(tf.truncated_normal([1, 1, n_hidden * 2, len(self.char_list) + 1], stddev=0.1))
        rnn = tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME')

        return tf.squeeze(rnn, axis=[2])

    def build_CTC(self, input_ctc):
        "create CTC loss and decoder and return them"

        # BxTxC -> TxBxC
        ctcIn3dTBC = tf.transpose(input_ctc, [1, 0, 2])
        # ground truth text as sparse tensor
        self.labels = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]),
                                      tf.placeholder(tf.int32, [None]),
                                      tf.placeholder(tf.int64, [2]))
        # calc loss for batch
        self.seq_length = tf.placeholder(tf.int32, [None])
        loss = tf.nn.ctc_loss(labels=self.labels,
                              inputs=ctcIn3dTBC,
                              sequence_length=self.seq_length,
                              ctc_merge_repeated=True)

        decoder = tf.nn.ctc_greedy_decoder(inputs=ctcIn3dTBC, sequence_length=self.seq_length)
        return (tf.reduce_mean(loss), decoder)

    def init_TensorFlow(self):

        sess = tf.Session()

        saver = tf.train.Saver()  # saver saves model to file
        latest_snapshot = tf.train.latest_checkpoint(self.path_model)  # is there a saved saved-model?

        if not latest_snapshot:
            print('Init with new values')
            sess.run(tf.global_variables_initializer())
        else:
            print('Init with stored values from ' + latest_snapshot)
            saver.restore(sess, latest_snapshot)

        return (sess, saver)

    def encode(self, texts):
        "transform labels to sparse tensor"

        indices = []
        values = []
        shape = [len(texts), 0]  # last entry must be max(labelList[i])

        for (batch_element, text) in enumerate(texts):
            # convert to string of label (i.e. class-ids)
            label_list = [self.char_list.index(c) for c in text]
            # sparse tensor must have size of max. label-string
            if len(label_list) > shape[1]:
                shape[1] = len(label_list)
            # put each label into sparse tensor
            for (i, label) in enumerate(label_list):
                indices.append([batch_element, i])
                values.append(label)

        return (indices, values, shape)

    def decode(self, ctc_output):
        "transform sparse tensor to labels"

        # ctc returns tuple, first element is SparseTensor
        decoded = ctc_output[0][0]

        # go over all indices and save mapping: batch -> values
        idxDict = {b: [] for b in range(self.batch_size)}
        encodedLabelStrs = [[] for i in range(self.batch_size)]

        for (idx, idx2d) in enumerate(decoded.indices):
            label = decoded.values[idx]
            batchElement = idx2d[0]  # index according to [b,t]
            encodedLabelStrs[batchElement].append(label)

        # map labels to chars for all batch elements
        return [str().join([self.char_list[c] for c in labelStr]) for labelStr in encodedLabelStrs]

    def train_batch(self, batch, batch_index):

        sparse = self.encode(batch.labels)
        train_data = {self.input_imgs: batch.imgs,
                      self.labels: sparse,
                      self.seq_length: [self.text_length] * self.batch_size}

        (_, lossVal) = self.sess.run([self.optimizer, self.loss], feed_dict=train_data)

        return lossVal

    def infer_batch(self, batch):

        infer_data = {self.input_imgs: batch.imgs,
                      self.seq_length: [self.text_length] * self.batch_size}

        decoded = self.sess.run(self.decoder, feed_dict=infer_data)
        return self.decode(decoded)

    def save(self, accuracy, epoch):

        # shutil.rmtree(self.path_model)
        # os.mkdir(self.path_model)

        file_name = self.path_model + 'Model-Epoch ' + str(epoch) + 'AC ' + str(round(accuracy, 2))
        self.saver.save(self.sess, file_name)
        print('Model Saved! ', file_name)
