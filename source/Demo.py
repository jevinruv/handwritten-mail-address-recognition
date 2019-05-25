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

        print("Model Loading Started")
        self.model = Model(open(self.file_char_list).read(), restore_model=True)
        print("Model Loading Finished")

    def recognize_text(self, test_img):
        word_list = []

        print("Image Processing Started")
        img_handler = ImageHandler()

        words = img_handler.split_text(test_img)

        for word in words:
            img = cv2.cvtColor(word, cv2.COLOR_BGR2GRAY)
            # img = img_handler.preprocess_normal_handwriting(img)
            img = img_handler.preprocess(img, self.img_size)
            word_list.append(img)
            # cv2.imshow('word', word)
            # cv2.waitKey(0)
        print("Image Processing Finished")

        print("Recognizing Text Started")
        batch = Batch(None, word_list)
        (recognized_list, probability) = self.model.batch_test(batch, True)
        # print('Image Text: ', recognized)

        text = ''
        for i in recognized_list:
            text += i + ' '

        print('Image Text: ' + text)
        print("\n")

    def recognize_all(self):

        for file_img in os.listdir(self.path_resources):
            if file_img.endswith(".png"):
                self.recognize_text(self.path_resources + file_img)

    def recognize_single(self):

        self.recognize_text(self.file_test_img)


demo = Demo()
demo.recognize_all()
# demo.recognize_single()

# img = cv2.imread('../resources/test1.png', cv2.IMREAD_GRAYSCALE)
# img = preprocess(img, Model.img_size)
# cv2.imshow('word', img)
# cv2.waitKey(0)
# plt.imshow(img, cmap='gray')
# plt.imshow(img)
# plt.show()
