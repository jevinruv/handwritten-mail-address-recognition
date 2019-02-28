import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from DataPrep import DataPrep
from Batch import Batch
from Model import Model
from ImagePreProcess import preprocess

fnCharList = '../model/charList.txt'
fnTrain = "../../../../../../Dataset/"
# fnTrain = "../../../Dataset/"
fnInfer = '../data/test.png'
n_epochs = 10


class Main:

    def __init__(self):

        # init class variables
        self.model = None
        self.loader = None

    def create_new_model(self):

        # load training data
        self.loader = DataPrep(fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)

        # create TF model
        self.model = Model(self.loader.charList)

        # save characters of model for inference mode
        open(fnCharList, 'w').write(str().join(self.loader.charList))

        for epoch in range(n_epochs):
            print('Epoch No. ', epoch, " of ", n_epochs)
            self.train()
            self.test()
            epoch += 1

        self.model.save()

    def train(self):
        print('Training Neural Network Started!')
        self.loader.trainSet()
        self.loader.shuffle()
        for _ in tqdm(range(int(self.loader.hasNext()))):
            v = self.loader.hasNext()
            iterInfo = self.loader.getIteratorInfo()
            batch = self.loader.getNext()
            loss = self.model.trainBatch(batch)
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
            # loss = self.model.trainBatch(batch)
            recognized = self.model.inferBatch(batch)

            # print('Ground truth -> Recognized')
            for i in range(len(recognized)):
                isOK = batch.gtTexts[i] == recognized[i]
                # print('[OK]' if isOK else '[ERR]', '"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
                numOK += 1 if isOK else 0
                numTotal += 1

        print(" corr ", numOK, " total ", numTotal)
        print('Correctly recognized words:', numOK / numTotal * 100.0, '%')

    def recognize_text(self):
        "recognize text in image provided by file path"

        self.model = Model(open(fnCharList).read())
        img = cv2.imread(fnInfer, cv2.IMREAD_GRAYSCALE)
        img = preprocess(img, Model.imgSize)
        batch = Batch(None, [img] * Model.batchSize)
        recognized = self.model.inferBatch(batch)
        print('Image Text: ', recognized[0])


main = Main()
main.create_new_model()
# main.infer()
