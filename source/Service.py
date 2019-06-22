import cv2

from Batch import Batch
from Constants import Constants
from ImageHandler import ImageHandler
from Model import Model


class Service:

    def __init__(self):
        self.path_resources = Constants.path_resources
        self.file_char_list = Constants.file_char_list
        self.file_test_img = Constants.file_test_img
        self.img_size = Constants.img_size
        self.batch_size = Constants.batch_size

        print("Model Loading Started")
        self.model = Model(open(self.file_char_list).read(), restore_model=True)
        print("Model Loading Finished")

    def recognize_text(self, img):
        word_list = []

        print("Image Processing Started")
        img_handler = ImageHandler()

        words = img_handler.split_text(img)

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

        return text
