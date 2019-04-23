import os
import cv2
import matplotlib.pyplot as plt

from Batch import Batch
from ImageHandler import preprocess, preprocess_normal_handwriting, split_text
from Model import Model

path_test_img = '../resources/'
file_char_list = '../resources/chars.txt'
file_test_img = '../resources/test0.png'


class Demo:
    def recognize_text(self):
        word_list = []

        print("Model Loading Started")
        model = Model(open(file_char_list).read())
        print("Model Loading Finished")

        print("Image Processing Started")
        img = cv2.imread(file_test_img)
        line_list = split_text(img, 'line')

        for line in line_list:
            line_segmented = split_text(line, 'word')

            for w in line_segmented:
                img = cv2.cvtColor(w, cv2.COLOR_BGR2GRAY)
                img = preprocess_normal_handwriting(img)
                img = preprocess(img, Model.img_size)
                word_list.append(img)
        print("Image Processing Finished")

        # for word in word_list:
        # cv2.imshow('word', word)
        # cv2.waitKey(0)
        # plt.imshow(word, cmap='gray')
        # plt.show()

        n_words = len(word_list)
        if (n_words < 50):
            sum = 50 - len(word_list)
            img = preprocess(None, Model.img_size)

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
    model = Model(open(file_char_list).read())

    for img_file in os.listdir(path_test_img):
        if img_file.endswith(".png"):
            img = cv2.imread(path_test_img + img_file, cv2.IMREAD_GRAYSCALE)
            img = preprocess(img, Model.img_size)
            batch = Batch(None, [img] * Model.batch_size)
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
