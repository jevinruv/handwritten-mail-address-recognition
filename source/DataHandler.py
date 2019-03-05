import random
import cv2

from Batch import Batch
from Sample import Sample
from ImageHandler import preprocess


class DataHandler:

    def __init__(self, file_path, batchSize, imgSize, maxTextLen):
        "loader for dataset at given location, preprocess images and text according to parameters"

        assert file_path[-1] == '/'

        self.currIdx = 0
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.samples = []
        self.file_path = file_path

        f = open(file_path + 'words.txt')
        chars = set()
        for line in f:
            # ignore comment line
            if not line or line[0] == '#':
                continue

            line_split = line.strip().split(' ')
            assert len(line_split) >= 9

            file_name = self.split_file_name(line_split)

            # GT text are columns starting at 9
            gtText = ' '.join(line_split[8:])[:maxTextLen]
            chars = chars.union(set(list(gtText)))

            # put sample into list
            self.samples.append(Sample(gtText, file_name))

        # split train 85% and test 15%
        splitIdx = int(0.85 * len(self.samples))
        self.trainSamples = self.samples[:splitIdx]
        self.validationSamples = self.samples[splitIdx:]

        # start with train set
        self.trainSet()

        # list of all chars in dataset
        self.charList = sorted(list(chars))

    def trainSet(self):
        "switch to training set"
        self.currIdx = 0
        self.samples = self.trainSamples

    def validationSet(self):
        "switch to validation set"
        self.currIdx = 0
        self.samples = self.validationSamples

    def shuffle(self):
        "shuffle current set"
        self.currIdx = 0
        random.shuffle(self.samples)

    def getIteratorInfo(self):
        "current batch index and overall number of batches"
        return (self.currIdx // self.batchSize, len(self.samples) // self.batchSize)

    def getBatchCount(self):
        return len(self.samples) / self.batchSize

    def split_file_name(self, line_split):
        # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
        file_name_split = line_split[0].split('-')

        file_name = self.file_path + 'words/' + \
                    file_name_split[0] + '/' + \
                    file_name_split[0] + '-' + \
                    file_name_split[1] + '/' + \
                    line_split[0] + '.png'
        return file_name

    # def split_dataset(self):

    # def hasNext(self):
    #     "iterator"
    #     return self.currIdx + self.batchSize <= len(self.samples)

    def getNext(self):
        "iterator"
        gtTexts = []
        imgs = []

        batchRange = range(self.currIdx, self.currIdx + self.batchSize)
        # gtTexts = [self.samples[i].gtText for i in batchRange]
        # imgs = [preprocess(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imgSize) for i in
        #         batchRange]

        for i in batchRange:
            gtTexts.append(self.samples[i].gtText)
            imgs.append(preprocess(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imgSize))

        self.currIdx += self.batchSize
        return Batch(gtTexts, imgs)