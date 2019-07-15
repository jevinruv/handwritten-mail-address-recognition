import argparse
import os
import cv2
import matplotlib.pyplot as plt
import textdistance

from Constants import Constants
from Service import Service


class Demo:

    def __init__(self):

        self.total_addresses = 0
        self.address_accuracy = 0
        self.total_addresses_by_type = 0
        self.address_accuracy_by_type = 0
        self.accuracy_list = []

        self.path_resources = Constants.path_resources
        self.file_test_img = Constants.file_test_img
        self.path_test_addresses = Constants.path_test_addresses
        self.path_test_address_file = Constants.path_test_address_file

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

        for folder in os.listdir(self.path_test_addresses):
            if not folder.endswith(".txt"):
                print('>> Start of Testing ' + folder)
                self.recognize_address_by_type(folder)
                print('>> End of Testing ' + folder)

        self.calculate_total_accuracy()

    def recognize_address_by_type(self, folder):

        address_test_file = self.path_test_address_file + folder + ".txt"

        f = open(address_test_file, "r")
        line_list = f.readlines()

        n_lines = len(line_list)

        for i in range(n_lines):
            file_name = self.path_test_addresses + folder + '/' + str(i) + ".png"
            print(file_name)

            img = cv2.imread(file_name)
            text_list = self.service.test_address(img)

            print(">> recognized")
            print(text_list)
            print(">> txt")
            print(line_list[i])

            self.calculate_address_accuracy(text_list, line_list[i])

        self.calculate_total_accuracy_per_type(folder)

    def calculate_address_accuracy(self, recognized_list, label):

        recognized = ''

        for i in recognized_list:
            recognized += i + ' '

        accuracy_address = textdistance.levenshtein.normalized_similarity(recognized, label)
        accuracy_address = (accuracy_address * 100)

        self.address_accuracy_by_type += accuracy_address
        self.total_addresses_by_type += 1

        print(">> Accuracy by Address " + str(accuracy_address) + "%")
        print("__________________________________________________________________________________________")

    def calculate_total_accuracy_per_type(self, address_type):

        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        accuracy_addresses = (self.address_accuracy_by_type / self.total_addresses_by_type)
        print(">> Type Address Accuracy " + str(accuracy_addresses) + "%")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        self.accuracy_list.append({address_type: str(accuracy_addresses)})
        self.address_accuracy += self.address_accuracy_by_type
        self.total_addresses += self.total_addresses_by_type
        self.address_accuracy_by_type = 0
        self.total_addresses_by_type = 0

    def calculate_total_accuracy(self):

        print("##########################################################################################")
        accuracy_addresses = (self.address_accuracy / self.total_addresses)
        print(">> Total Address Accuracy " + str(accuracy_addresses) + "%")

        for address in self.accuracy_list:
            print(address)
        print("##########################################################################################")


demo = Demo()

parser = argparse.ArgumentParser()
parser.add_argument('--address', help='test all generated', action='store_true')
# parser.add_argument('--all', help='test all in resource folder', action='store_true')

args = parser.parse_args()

if args.address:
    demo.recognize_addresses()
else:
    demo.recognize_all()
