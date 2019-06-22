import os
import cv2
import matplotlib.pyplot as plt

from Constants import Constants
from Service import Service


class Demo:

    def __init__(self):
        self.path_resources = Constants.path_resources
        self.file_test_img = Constants.file_test_img

        self.service = Service()

    def recognize_all(self):

        for file_img in os.listdir(self.path_resources):
            if file_img.endswith(".png"):
                img = cv2.imread(self.path_resources + file_img)
                text = self.service.recognize_text(img)
                print(text)

    def recognize_single(self):

        img = cv2.imread(self.file_test_img)
        text = self.service.recognize_text(img)
        print(text)


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
