import sys
import numpy as np
import tensorflow as tf

from Constants import Constants
from DataHandler import DataHandler


class Model:

    def __init__(self, char_list, restore_model=False):

        self.mustRestore = restore_model
        self.snapshot_id = 0

        self.decoder_selected = Constants.decoder_selected
        self.path_model = Constants.path_model
        self.batch_size = Constants.batch_size
        self.char_list = char_list
        self.learning_rate = Constants.learning_rate
        self.text_length = Constants.text_length
        self.img_size = Constants.img_size
        self.file_word_char_list = Constants.file_word_char_list
        self.file_word_beam_search = Constants.file_word_beam_search
        # self.file_corpus = Constants.file_corpus
        self.file_collection_words = Constants.file_collection_words

        # Whether to use normalization over a batch or a population
        self.is_train = tf.placeholder(tf.bool, name='is_train')

        # input image batch
        self.input_images = tf.placeholder(tf.float32, shape=(None, self.img_size[0], self.img_size[1]))

        self.build_CNN()
        self.build_RNN()
        self.build_CTC()

        # setup optimizer to train NN
        self.batchesTrained = 0
        self.learningRate = tf.placeholder(tf.float32, shape=[])
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)

        # initialize TF
        (self.sess, self.saver) = self.build_TF()

    def build_CNN(self):

        input_4d = tf.expand_dims(input=self.input_images, axis=3)  # adds dimensions of size 1 to the 3rd index

        pool = input_4d

        pool = self.create_CNN_layer(pool, filter_size=5, in_features=1, out_features=32, max_pool=(2, 2))
        pool = self.create_CNN_layer(pool, filter_size=5, in_features=32, out_features=64, max_pool=(2, 2))
        pool = self.create_CNN_layer(pool, filter_size=3, in_features=64, out_features=128, max_pool=(1, 2))
        pool = self.create_CNN_layer(pool, filter_size=3, in_features=128, out_features=128, max_pool=(1, 2))
        pool = self.create_CNN_layer(pool, filter_size=3, in_features=128, out_features=256, max_pool=(1, 2))

        self.cnnOut4d = pool

    def create_CNN_layer(self, pool, filter_size, in_features, out_features, max_pool):

        # initialize weights
        filter = tf.Variable(tf.truncated_normal([filter_size, filter_size, in_features, out_features], stddev=0.1))

        conv = tf.nn.conv2d(pool, filter, padding='SAME', strides=(1, 1, 1, 1))
        conv_norm = tf.layers.batch_normalization(conv, training=self.is_train)
        relu = tf.nn.relu(conv_norm)
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

    def build_RNN(self):
        "create RNN layers and return output of these layers"
        rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])

        # basic cells which is used to build RNN
        numHidden = 256
        cells = [tf.contrib.rnn.LSTMCell(num_units=numHidden, state_is_tuple=True) for _ in range(2)]  # 2 layers

        # stack basic cells
        stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        # bidirectional RNN
        # BxTxF -> BxTx2H
        ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d,
                                                        dtype=rnnIn3d.dtype)

        # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

        # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden * 2, len(self.char_list) + 1], stddev=0.1))
        rnn = tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME')

        self.rnnOut3d = tf.squeeze(rnn, axis=[2])

    def build_CTC(self):
        "create CTC loss and decoder and return them"
        # BxTxC -> TxBxC
        self.ctcIn3dTBC = tf.transpose(self.rnnOut3d, [1, 0, 2])
        # ground truth text as sparse tensor
        self.labels = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]),
                                      tf.placeholder(tf.int32, [None]),
                                      tf.placeholder(tf.int64, [2]))

        # calc loss for batch
        self.seq_length = tf.placeholder(tf.int32, [None])
        self.loss = tf.reduce_mean(
            tf.nn.ctc_loss(labels=self.labels,
                           inputs=self.ctcIn3dTBC,
                           sequence_length=self.seq_length,
                           ctc_merge_repeated=True))

        # calc loss for each element to compute label probability
        self.savedCtcInput = tf.placeholder(tf.float32, shape=[self.text_length, None, len(self.char_list) + 1])
        self.lossPerElement = tf.nn.ctc_loss(labels=self.labels, inputs=self.savedCtcInput,
                                             sequence_length=self.seq_length, ctc_merge_repeated=True)

        # decoder: either best path decoding or beam search decoding
        if self.decoder_selected == Constants.decoder_best_path:
            self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seq_length)

        elif self.decoder_selected == Constants.decoder_word_beam:
            print(">>>>>>>>>>>>>>>>>> Word Beam")
            self.load_word_beam()

    def load_word_beam(self):

        word_beam_search_module = tf.load_op_library(self.file_word_beam_search)

        chars = str().join(self.char_list)
        word_chars = open(self.file_word_char_list).read().splitlines()[0]

        data_handler = DataHandler()
        data_handler.prepare_collection_words()
        collection_words = open(self.file_collection_words).read()

        # decode using the "Words" mode of word beam search
        self.decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(self.ctcIn3dTBC, dim=2),
                                                                50,
                                                                'Words',
                                                                0.0,
                                                                collection_words.encode('utf8'),
                                                                chars.encode('utf8'),
                                                                word_chars.encode('utf8'))

    def build_TF(self):
        "initialize TF"

        sess = tf.Session()

        saver = tf.train.Saver(max_to_keep=1)  # saver saves model to file
        latestSnapshot = tf.train.latest_checkpoint(self.path_model)  # is there a saved model?

        # if model must be restored (for inference), there must be a snapshot
        if self.mustRestore and not latestSnapshot:
            raise Exception('Model Not found')

        # load saved model if available
        if latestSnapshot:
            print('Init with stored values from ' + latestSnapshot)
            saver.restore(sess, latestSnapshot)
        else:
            print('Init with new values')
            sess.run(tf.global_variables_initializer())

        return (sess, saver)

    def encode(self, texts):
        "put ground truth texts into sparse tensor for ctc_loss"

        indices = []
        values = []
        shape = [len(texts), 0]  # last entry must be max(labelList[i])

        # go over all texts
        for (batchElement, text) in enumerate(texts):
            # convert to string of label (i.e. class-ids)
            labelStr = [self.char_list.index(c) for c in text]
            # sparse tensor must have size of max. label-string
            if len(labelStr) > shape[1]:
                shape[1] = len(labelStr)
            # put each label into sparse tensor
            for (i, label) in enumerate(labelStr):
                indices.append([batchElement, i])
                values.append(label)

        return (indices, values, shape)

    def decode(self, ctcOutput, batchSize):
        "extract texts from output of CTC decoder"

        # contains string of labels for each batch element
        encodedLabelStrs = [[] for i in range(batchSize)]

        # word beam search: label strings terminated by blank
        if self.decoder_selected == Constants.decoder_word_beam:
            blank = len(self.char_list)
            for b in range(batchSize):
                for label in ctcOutput[b]:
                    if label == blank:
                        break
                    encodedLabelStrs[b].append(label)

        # TF decoders: label strings are contained in sparse tensor
        else:
            # ctc returns tuple, first element is SparseTensor
            decoded = ctcOutput[0][0]

            # go over all indices and save mapping: batch -> values
            idxDict = {b: [] for b in range(batchSize)}
            for (idx, idx2d) in enumerate(decoded.indices):
                label = decoded.values[idx]
                batchElement = idx2d[0]  # index according to [b,t]
                encodedLabelStrs[batchElement].append(label)

        # map labels to chars for all batch elements
        return [str().join([self.char_list[c] for c in labelStr]) for labelStr in encodedLabelStrs]

    def batch_train(self, batch):
        "feed a batch into the NN to train it"

        numBatchElements = len(batch.imgs)
        sparse = self.encode(batch.labels)

        rate = 0.01 if self.batchesTrained < 10 else (
            0.001 if self.batchesTrained < 10000 else 0.0001)  # decay learning rate

        evalList = [self.optimizer, self.loss]

        data_train = {self.input_images: batch.imgs,
                      self.labels: sparse,
                      self.seq_length: [self.text_length] * numBatchElements,
                      self.learningRate: rate,
                      self.is_train: True}

        (_, lossVal) = self.sess.run(evalList, data_train)
        self.batchesTrained += 1

        return lossVal

    def batch_test(self, batch, calcProbability=False, probabilityOfGT=False):
        "feed a batch into the NN to recognize the texts"

        # decode, optionally save RNN output
        numBatchElements = len(batch.imgs)
        evalList = [self.decoder] + ([self.ctcIn3dTBC] if calcProbability else [])

        data_test = {self.input_images: batch.imgs,
                     self.seq_length: [self.text_length] * numBatchElements,
                     self.is_train: False}

        evalRes = self.sess.run([self.decoder, self.ctcIn3dTBC], data_test)

        decoded = evalRes[0]
        texts = self.decode(decoded, numBatchElements)

        # feed RNN output and recognized text into CTC loss to compute labeling probability
        probs = None
        if calcProbability:
            sparse = self.encode(batch.labels) if probabilityOfGT else self.encode(texts)
            ctcInput = evalRes[1]

            evalList = self.lossPerElement

            data_test = {self.savedCtcInput: ctcInput,
                         self.labels: sparse,
                         self.seq_length: [self.text_length] * numBatchElements,
                         self.is_train: False}

            lossVals = self.sess.run(evalList, data_test)
            probs = np.exp(-lossVals)

        return (texts, probs)

    def save(self):
        "save model to file"
        self.snapshot_id += 1
        self.saver.save(self.sess, '../saved-model/snapshot', global_step=self.snapshot_id)
