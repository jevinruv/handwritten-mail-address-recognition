import cv2
import matplotlib.pyplot as plt
from DataLoader import DataLoader
from Batch import Batch
from Model import Model
from SamplePreprocessor import preprocess

# filenames and paths to data
fnCharList = '../model/charList.txt'
# fnTrain = '../data/'
fnTrain = "../../../Dataset/"
fnInfer = '../data/test.png'
n_epochs = 3


def train():
    "train NN"

    # load training data
    loader = DataLoader(fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)

    # create TF model
    model = Model(loader.charList)

    # save characters of model for inference mode
    open(fnCharList, 'w').write(str().join(loader.charList))

    for epoch in range(n_epochs):
        print('Epoch:', epoch)
        model.save()

        # train
        print('Train NN')
        loader.trainSet()
        loader.shuffle()
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch)
            # print('Iterator:', iterInfo, 'Loss:', loss)

        # validate
        print('Validate NN')
        loader.validationSet()
        numOK = 0
        numTotal = 0
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            # print('Iterator:', iterInfo)
            batch = loader.getNext()
            loss = model.trainBatch(batch)
            recognized = model.inferBatch(batch)

            print('Ground truth -> Recognized')
            for i in range(len(recognized)):
                isOK = batch.gtTexts[i] == recognized[i]
                # print('[OK]' if isOK else '[ERR]', '"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
                numOK += 1 if isOK else 0
                numTotal += 1
        # print validation result
        print('Correctly recognized words:', numOK / numTotal * 100.0, '%')

        epoch += 1


def infer():
    "recognize text in image provided by file path"

    model = Model(open(fnCharList).read())
    img = cv2.imread(fnInfer, cv2.IMREAD_GRAYSCALE)
    img = preprocess(img, Model.imgSize)
    batch = Batch(None, [img] * Model.batchSize)
    recognized = model.inferBatch(batch)
    print('Image Text: ', recognized[0])


if __name__ == '__main__':
    train()
    # infer()
