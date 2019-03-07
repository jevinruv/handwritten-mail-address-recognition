import tensorflow as tf
import os
import shutil


class Model:
    # constants
    batch_size = 50
    img_size = (128, 32)
    maxTextLen = 32
    learning_rate = 0.0001
    path_model = '../saved-model/'

    def __init__(self, char_list):
        "init saved-model: add CNN, RNN and CTC and initialize TF"

        self.char_list = char_list
        self.snapID = 0

        # CNN
        self.input_imgs = tf.placeholder(tf.float32, shape=(Model.batch_size, Model.img_size[0], Model.img_size[1]))
        # self.inputImgs = tf.placeholder(tf.float32, shape=(Model.batchSize, self.IMG_WIDTH, self.IMG_HEIGHT))
        cnnOut4d = self.build_CNN(self.input_imgs)

        # RNN
        rnnOut3d = self.build_RNN(cnnOut4d)

        # CTC
        (self.loss, self.decoder) = self.build_CTC(rnnOut3d)

        # optimizer for NN parameters
        # self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # initialize TF
        (self.sess, self.saver) = self.setupTF()

    def build_CNN(self, cnnIn3d):
        "create CNN layers and return output of these layers"

        cnnIn4d = tf.expand_dims(input=cnnIn3d, axis=3)

        # list of parameters for the layers
        filter_size = [5, 5, 3, 3, 3]
        feature_values = [1, 32, 64, 128, 128, 256]
        strideVals = poolVals = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
        n_layers = len(strideVals)

        # create layers
        pool = cnnIn4d

        for i in range(n_layers):
            filter = tf.Variable(
                tf.truncated_normal([filter_size[i], filter_size[i], feature_values[i], feature_values[i + 1]],
                                    stddev=0.1))
            conv = tf.nn.conv2d(pool, filter, padding='SAME', strides=(1, 1, 1, 1))
            relu = tf.nn.relu(conv)
            pool = tf.nn.max_pool(relu,
                                  ksize=(1, poolVals[i][0], poolVals[i][1], 1),
                                  strides=(1, strideVals[i][0], strideVals[i][1], 1),
                                  padding='VALID')
        # layer 1
        # filter = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
        # conv = tf.nn.conv2d(input=pool, filter=filter, padding='SAME', strides=(1, 1, 1, 1))
        # relu = tf.nn.relu(conv)
        # pool = tf.nn.max_pool(relu, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID')

        return pool

    def build_RNN(self, rnnIn4d):

        rnnIn3d = tf.squeeze(rnnIn4d, axis=[2])  # squeeze remove 1 dimensions, here it removes the 2nd index

        n_hidden = 256
        n_layers = 2

        cells = []
        for _ in range(n_layers):
            cells.append(tf.nn.rnn_cell.LSTMCell(num_units=n_hidden))

        stacked = tf.nn.rnn_cell.MultiRNNCell(cells)    # combine the 2 LSTMCell created

        # BxTxF -> BxTx2H
        ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d,
                                                        dtype=rnnIn3d.dtype)

        # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

        # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        kernel = tf.Variable(tf.truncated_normal([1, 1, n_hidden * 2, len(self.char_list) + 1], stddev=0.1))
        rnn = tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME')

        return tf.squeeze(rnn, axis=[2])

    def build_CTC(self, ctcIn3d):
        "create CTC loss and decoder and return them"
        # BxTxC -> TxBxC
        ctcIn3dTBC = tf.transpose(ctcIn3d, [1, 0, 2])
        # ground truth text as sparse tensor
        self.labels = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]),
                                      tf.placeholder(tf.int32, [None]),
                                      tf.placeholder(tf.int64, [2]))
        # calc loss for batch
        self.seqLen = tf.placeholder(tf.int32, [None])
        loss = tf.nn.ctc_loss(labels=self.labels, inputs=ctcIn3dTBC, sequence_length=self.seqLen,
                              ctc_merge_repeated=True)

        decoder = tf.nn.ctc_greedy_decoder(inputs=ctcIn3dTBC, sequence_length=self.seqLen)
        return (tf.reduce_mean(loss), decoder)

    def setupTF(self):
        "initialize TF"

        sess = tf.Session()

        saver = tf.train.Saver()  # saver saves saved-model to file
        latestSnapshot = tf.train.latest_checkpoint('../saved-model/')  # is there a saved saved-model?

        # no saved saved-model -> init with new values
        if not latestSnapshot:
            print('Init with new values')
            sess.run(tf.global_variables_initializer())
        # init with saved values
        else:
            print('Init with stored values from ' + latestSnapshot)
            saver.restore(sess, latestSnapshot)

        return (sess, saver)

    def toSparse(self, texts):
        "transfor labels into sparse tensor for ctc_loss"

        indices = []
        values = []
        shape = [len(texts), 0]  # last entry must be max(labelList[i])

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

    def fromSparse(self, ctcOutput):
        "extract texts from sparse tensor"
        # ctc returns tuple, first element is SparseTensor
        decoded = ctcOutput[0][0]

        # go over all indices and save mapping: batch -> values
        idxDict = {b: [] for b in range(Model.batch_size)}
        encodedLabelStrs = [[] for i in range(Model.batch_size)]
        for (idx, idx2d) in enumerate(decoded.indices):
            label = decoded.values[idx]
            batchElement = idx2d[0]  # index according to [b,t]
            encodedLabelStrs[batchElement].append(label)

        # map labels to chars for all batch elements
        return [str().join([self.char_list[c] for c in labelStr]) for labelStr in encodedLabelStrs]

    def train_batch(self, batch):
        "feed a batch into the NN to train it"

        sparse = self.toSparse(batch.gtTexts)
        train_data = {self.input_imgs: batch.imgs, self.labels: sparse,
                      self.seqLen: [Model.maxTextLen] * Model.batch_size}

        (_, lossVal) = self.sess.run([self.optimizer, self.loss], feed_dict=train_data)
        return lossVal

    def inferBatch(self, batch):
        "feed a batch into the NN to recngnize the texts"

        decoded = self.sess.run(self.decoder,
                                {self.input_imgs: batch.imgs,
                                 self.seqLen: [Model.maxTextLen] * Model.batch_size})
        return self.fromSparse(decoded)

    def save(self, accuracy):

        shutil.rmtree(self.path_model)
        os.mkdir(self.path_model)

        file_name = self.path_model + 'snapshot-acc ' + str(round(accuracy, 2))
        self.saver.save(self.sess, file_name)
        print('Model Saved! ', file_name)
