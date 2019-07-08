import argparse
import os
import cv2
import matplotlib.pyplot as plt
import textdistance

from Constants import Constants
from Service import Service


class Demo:

    def __init__(self):

        self.correct_words = 0
        self.correct_address = 0

        self.total_words = 0
        self.total_addresses = 0

        self.path_resources = Constants.path_resources
        self.file_test_img = Constants.file_test_img
        self.path_test_addresses = Constants.path_test_addresses
        self.file_collection_test_address = Constants.file_collection_test_address

        self.service = Service()

    def recognize_all(self):

        ext = [".png", "jpg"]

        for file_img in os.listdir(self.path_resources):
            if file_img.endswith(tuple(ext)):
                img = cv2.imread(self.path_resources + file_img)
                text = self.service.recognize_text(img)
                # print(text)

    def recognize_single(self):

        img = cv2.imread(self.file_test_img)
        text = self.service.recognize_text(img)
        print(text)

    def recognize_addresses(self):

        f = open(self.file_collection_test_address, "r")
        line_list = f.readlines()

        n_lines = len(line_list)

        for i in range(n_lines):
            file_name = self.path_test_addresses + str(i + 1) + ".png"
            print(file_name)

            img = cv2.imread(file_name)
            text_list = self.service.test_address(img)

            print(">> recognized")
            print(text_list)
            print(">> txt")
            print(line_list[i])

            self.calculate_address_accuracy(text_list, line_list[i])

        self.calculate_total_accuracy()

    def calculate_address_accuracy(self, recognized_list, label):

        # start of address accuracy

        recognized = ''
        is_address_accurate = False

        self.total_addresses += 1

        for i in recognized_list:
            recognized += i + ' '

        if recognized == label:
            self.correct_address += 1
            is_address_accurate = True

        accuracy_add = textdistance.levenshtein.normalized_similarity(recognized, label)
        print(">> Accuracy by Algorithm " + str(accuracy_add))
        print(">> Accuracy by direct " + str(is_address_accurate))
        print("____________________________________________________________________________________________")

        # end of address accuracy

        # start of address word accuracy

        label_list = label.split()

        recognized_count = len(recognized_list)
        label_count = len(label_list)

        self.total_words += label_count

        correct_words_in_address = 0

        for i_label in label_list:

            for i_recognized in recognized_list:

                if i_label == i_recognized:
                    correct_words_in_address += 1

        self.correct_words += correct_words_in_address

        # end of address word accuracy

    def calculate_total_accuracy(self):

        print("##########################################################################################")
        accuracy_words = (self.correct_words / self.total_words) * 100
        accuracy_addresses = (self.correct_address / self.total_addresses) * 100
        print(">> " + str(accuracy_words) + "%")
        print(">> " + str(accuracy_addresses) + "%")
        print("###########################################################################################")


demo = Demo()

parser = argparse.ArgumentParser()
parser.add_argument('--address', help='test all generated', action='store_true')
# parser.add_argument('--all', help='test all in resource folder', action='store_true')

args = parser.parse_args()

if args.address:
    demo.recognize_addresses()
else:
    demo.recognize_all()

# img = cv2.imread('../resources/test1.png', cv2.IMREAD_GRAYSCALE)
# img = preprocess(img, Model.img_size)
# cv2.imshow('word', img)
# cv2.waitKey(0)
# plt.imshow(img, cmap='gray')
# plt.imshow(img)
# plt.show()
