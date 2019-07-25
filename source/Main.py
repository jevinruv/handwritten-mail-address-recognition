from tqdm import tqdm

from DataHandler import DataHandler
from Model import Model
from Constants import Constants


class Main:

    def __init__(self):

        self.model = None
        self.data_handler = None
        self.char_list = Constants.file_char_list
        self.collection_handwritten_words = Constants.file_collection_handwritten_words
        self.num_epochs = Constants.num_epochs
        self.top_accuracy = 0

    def create_new_model(self):

        self.data_handler = DataHandler()

        file_char_list = open(self.char_list, 'w')
        str_char_list = str().join(self.data_handler.char_list)
        file_char_list.write(str_char_list)

        file_collection_handwritten_words = open(self.collection_handwritten_words, 'w')
        str_collection_handwritten_words = str(' ').join(self.data_handler.words_train + self.data_handler.words_test)
        file_collection_handwritten_words.write(str_collection_handwritten_words)

        self.model = Model(self.data_handler.char_list)

        for epoch in range(self.num_epochs):
            print('Epoch ', epoch, ' of ', self.num_epochs)
            self.train()
            accuracy = self.test()
            self.model.save()

            # if self.top_accuracy < accuracy:
            #     self.top_accuracy = accuracy
            #     self.model.save(accuracy)

    def train(self):

        print('Training Neural Network Started!')

        self.data_handler.set_dataset('train')
        self.data_handler.shuffle()
        n_batch = int(self.data_handler.get_batch_count())

        for batch_index in tqdm(range(n_batch)):
            batch = self.data_handler.get_next()
            loss = self.model.batch_train(batch)

        print('Training Neural Network Ended!')

    def test(self):

        print('Testing Neural Network Started!')

        self.data_handler.set_dataset('test')
        self.data_handler.shuffle()
        n_correct = 0
        n_total = 0

        for _ in tqdm(range(int(self.data_handler.get_batch_count()))):

            batch = self.data_handler.get_next()
            recognized = self.model.batch_test(batch)

            for i in range(len(recognized)):
                is_correct = batch.labels[i] == recognized[i]
                n_correct += 1 if is_correct else 0
                n_total += 1

        accuracy = self.calculate_accuracy(n_total, n_correct)
        print('Testing Neural Network Ended!')

        return accuracy

    def calculate_accuracy(self, n_total, n_correct):

        print("Correct ", n_correct, " total ", n_total)
        accuracy = n_correct / n_total * 100.0
        print(accuracy, '% Correctly recognized words')

        return accuracy


main = Main()
main.create_new_model()
