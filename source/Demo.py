import os
import cv2
import matplotlib.pyplot as plt

from Constants import Constants
from Batch import Batch
from ImageHandler import ImageHandler
from Model import Model


class Demo:

    def __init__(self):
        self.path_resources = Constants.path_resources
        self.file_char_list = Constants.file_char_list
        self.file_test_img = Constants.file_test_img
        self.img_size = Constants.img_size
        self.batch_size = Constants.batch_size

    def recognize_text(self):
        word_list = []

        print("Model Loading Started")
        model = Model(open(self.file_char_list).read())
        # model = Model()
        print("Model Loading Finished")

        print("Image Processing Started")
        img_handler = ImageHandler()
        img = cv2.imread(self.file_test_img)
        line_list = img_handler.split_text(img, 'line')

        for line in reversed(line_list):
            line_segmented = img_handler.split_text(line, 'word')
            # cv2.imshow('line', line)
            # cv2.waitKey(0)

            for word in line_segmented:
                img = cv2.cvtColor(word, cv2.COLOR_BGR2GRAY)
                # img = img_handler.preprocess_normal_handwriting(img)
                img = img_handler.preprocess(img, self.img_size)
                word_list.append(img)
                # cv2.imshow('word', word)
                # cv2.waitKey(0)
        print("Image Processing Finished")

        # for w in line_list:
        #     cv2.imshow('word', w)
        #     cv2.waitKey(0)
            # plt.imshow(w, cmap='gray')
            # plt.show()

        n_words = len(word_list)
        if (n_words < 50):
            sum = 50 - len(word_list)
            img = img_handler.preprocess(None, self.img_size)

            for _ in range(sum):
                word_list.append(img)

        print("Recognizing Text Started")
        batch = Batch(None, word_list)
        recognized_list = model.infer_batch(batch)
        # print('Image Text: ', recognized)

        text = ''
        for i in range(n_words):
            text += recognized_list[i] + ' '

        print('Image Text: ' + text)


def test_extension(self):
    # model = Model()
    model = Model(open(self.file_char_list).read())

    for img_file in os.listdir(self.path_resources):
        if img_file.endswith(".png"):
            img_handler = ImageHandler()
            img = cv2.imread(self.path_resources + img_file, cv2.IMREAD_GRAYSCALE)
            img = img_handler.preprocess(img, self.img_size)
            batch = Batch(None, [img] * self.batch_size)
            recognized = model.infer_batch(batch)
            plt.imshow(img)
            plt.show()
            print('Image Text: ', recognized[0])


demo = Demo()
# demo.test_extension()
demo.recognize_text()

# img = cv2.imread('../resources/test1.png', cv2.IMREAD_GRAYSCALE)
# img = preprocess(img, Model.img_size)
# cv2.imshow('word', img)
# cv2.waitKey(0)
# plt.imshow(img, cmap='gray')
# plt.imshow(img)
# plt.show()
