import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from DataPrep import DataPrep
from Batch import Batch
from Model import Model
from ImagePreProcess import preprocess

path_char_list = '../resources/char-list.txt'
path_dataset = "../../../../../../Dataset/"
# fnTrain = "../../../Dataset/"
fnInfer = '../resources/test.png'
path_test_img = '../resources/'
n_epochs = 5


class Main:

    def __init__(self):

        # init class variables
        self.model = None
        self.loader = None
        self.top_accuracy = 0

    def create_new_model(self):

        # load training resources
        self.loader = DataPrep(path_dataset, Model.batchSize, Model.imgSize, Model.maxTextLen)

        # create TF saved-model
        self.model = Model(self.loader.charList)

        # save characters of saved-model for inference mode
        open(path_char_list, 'w').write(str().join(self.loader.charList))

        for epoch in range(n_epochs):
            print('Epoch ', epoch, ' of ', n_epochs)
            self.train()
            accuracy = self.test()
            if self.top_accuracy < accuracy:
                self.top_accuracy = accuracy
                self.model.save(accuracy)
            epoch += 1

    def train(self):
        print('Training Neural Network Started!')
        self.loader.trainSet()
        self.loader.shuffle()
        for _ in tqdm(range(int(self.loader.hasNext()))):
            v = self.loader.hasNext()
            iterInfo = self.loader.getIteratorInfo()
            batch = self.loader.getNext()
            loss = self.model.train_batch(batch)
            # print('Iterator:', iterInfo, 'Loss:', loss)

    def test(self):
        print('Testing Neural Network Started!')
        self.loader.validationSet()
        numOK = 0
        numTotal = 0

        for _ in tqdm(range(int(self.loader.hasNext()))):
            iterInfo = self.loader.getIteratorInfo()
            # print('Iterator:', iterInfo)
            batch = self.loader.getNext()
            # loss = self.saved-model.trainBatch(batch)
            recognized = self.model.inferBatch(batch)

            # print('Ground truth -> Recognized')
            for i in range(len(recognized)):
                isOK = batch.gtTexts[i] == recognized[i]
                # print('[OK]' if isOK else '[ERR]', '"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
                numOK += 1 if isOK else 0
                numTotal += 1

        print(" corr ", numOK, " total ", numTotal)
        accuracy = numOK / numTotal * 100.0
        print('Correctly recognized words:', accuracy, '%')
        return accuracy

    def recognize_text(self):
        "recognize text in image provided by file path"

        self.model = Model(open(path_char_list).read())
        img = cv2.imread(fnInfer, cv2.IMREAD_GRAYSCALE)
        img = preprocess(img, Model.imgSize)
        batch = Batch(None, [img] * Model.batchSize)
        recognized = self.model.inferBatch(batch)
        print('Image Text: ', recognized[0])

    def recognize_text_data(self):

        # self.model = None
        # self.model = Model(open(path_char_list).read())

        for img_file in os.listdir(path_test_img):
            if img_file.endswith(".png"):
                img = cv2.imread(path_test_img + img_file, cv2.IMREAD_GRAYSCALE)
                img = preprocess(img, Model.imgSize)
                batch = Batch(None, [img] * Model.batchSize)
                recognized = self.model.inferBatch(batch)
                plt.imshow(img)
                plt.show()
                print('Image Text: ', recognized[0])


main = Main()
main.create_new_model()
main.recognize_text_data()
