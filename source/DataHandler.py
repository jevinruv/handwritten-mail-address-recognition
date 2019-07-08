import random
import cv2

from Constants import Constants
from Batch import Batch
from ImageHandler import ImageHandler
from ImageInfo import ImageInfo


class DataHandler:

    def __init__(self):

        self.batch_size = Constants.batch_size
        self.img_size = Constants.img_size
        self.file_path = Constants.path_dataset
        self.text_length = Constants.text_length

        self.file_collection_handwritten_words = Constants.file_collection_handwritten_words
        self.file_collection_test_address = Constants.file_collection_test_address
        self.file_collection_address = Constants.file_collection_address
        self.collection_words = Constants.file_collection_words

        self.samples = []
        self.current_index = 0

        f = open(self.file_path + 'words.txt')
        chars = set()

        for line in f:

            # skip comment line
            if not line or line[0] == '#':
                continue

            line_split = line.strip().split(' ')
            assert len(line_split) >= 9

            file_name = self.split_file_name(line_split)

            # label is the column starting at 9
            label = ' '.join(line_split[8:])[:self.text_length]
            chars = chars.union(set(list(label)))

            self.samples.append(ImageInfo(label, file_name))

        # split train 85% and test 15%
        split_index = int(0.85 * len(self.samples))
        self.train_samples = self.samples[:split_index]
        self.test_samples = self.samples[split_index:]

        # put words into lists
        self.words_train = [x.label for x in self.train_samples]
        self.words_test = [x.label for x in self.test_samples]

        # default dataset
        self.set_train_data()

        # list of all chars in dataset
        self.char_list = sorted(list(chars))

    def set_train_data(self):
        self.current_index = 0
        self.samples = self.train_samples

    def set_test_data(self):
        self.current_index = 0
        self.samples = self.test_samples

    def shuffle(self):
        self.current_index = 0
        random.shuffle(self.samples)

    def get_batch_count(self):
        return len(self.samples) / self.batch_size

    def split_file_name(self, line_split):
        # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
        file_name_split = line_split[0].split('-')

        file_name = self.file_path + 'words/' + \
                    file_name_split[0] + '/' + \
                    file_name_split[0] + '-' + \
                    file_name_split[1] + '/' + \
                    line_split[0] + '.png'
        return file_name

    def get_next(self):

        labels = []
        imgs = []

        batch_range = range(self.current_index, self.current_index + self.batch_size)

        for i in batch_range:
            labels.append(self.samples[i].label)
            img = cv2.imread(self.samples[i].file_path, cv2.IMREAD_GRAYSCALE)
            img_handler = ImageHandler()
            img = img_handler.preprocess(img, self.img_size)
            imgs.append(img)

        self.current_index += self.batch_size
        return Batch(labels, imgs)

    def prepare_collection_words(self):

        file1 = open(self.file_collection_test_address, 'r')
        lines = file1.readlines()
        addresses_test = ' '.join([line.strip() for line in lines])
        file1.close()

        file2 = open(self.file_collection_address, 'r')
        lines2 = file2.readlines()
        addresses = ' '.join([line.strip() for line in lines2])
        file2.close()

        collection = addresses + addresses_test

        file3 = open(self.collection_words, "w")
        file3.write(collection)
        file3.close()
