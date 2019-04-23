import random
import cv2

from Batch import Batch
from ImageHandler import ImageHandler
from ImageInfo import ImageInfo


class DataHandler:

    def __init__(self, file_path, batchSize, imgSize, maxTextLen):

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

            # label is the column starting at 9
            label = ' '.join(line_split[8:])[:maxTextLen]
            chars = chars.union(set(list(label)))

            # put sample into list
            self.samples.append(ImageInfo(label, file_name))

        # split train 85% and test 15%
        split_index = int(0.85 * len(self.samples))
        self.trainSamples = self.samples[:split_index]
        self.validationSamples = self.samples[split_index:]

        # default dataset
        self.set_train_data()

        # list of all chars in dataset
        self.charList = sorted(list(chars))

    def set_train_data(self):
        self.currIdx = 0
        self.samples = self.trainSamples

    def set_test_data(self):
        self.currIdx = 0
        self.samples = self.validationSamples

    def shuffle(self):
        self.currIdx = 0
        random.shuffle(self.samples)

    def getIteratorInfo(self):
        return (self.currIdx // self.batchSize, len(self.samples) // self.batchSize)

    def get_batch_count(self):
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

    # def hasNext(self):
    #     return self.currIdx + self.batchSize <= len(self.samples)

    def get_next(self):

        labels = []
        imgs = []

        batch_range = range(self.currIdx, self.currIdx + self.batchSize)

        for i in batch_range:
            labels.append(self.samples[i].label)
            img = cv2.imread(self.samples[i].file_path, cv2.IMREAD_GRAYSCALE)
            img_handler = ImageHandler()
            img = img_handler.preprocess(img, self.imgSize)
            imgs.append(img)

        self.currIdx += self.batchSize
        return Batch(labels, imgs)
