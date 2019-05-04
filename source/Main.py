from tqdm import tqdm

from DataHandler import DataHandler
from Model import Model
from Constants import Constants


class Main:

    def __init__(self):

        # init class variables
        self.model = None
        self.data_handler = None
        self.char_list = Constants.file_char_list
        self.epochs = Constants.epochs
        self.top_accuracy = 0

    def create_new_model(self):

        self.data_handler = DataHandler()
        # self.model = Model()
        self.model = Model(self.data_handler.char_list)

        # save characters of model for inference mode
        open(self.char_list, 'w').write(str().join(self.data_handler.char_list))

        for epoch in range(self.epochs):
            print('Epoch ', epoch, ' of ', self.epochs)
            self.train()
            accuracy = self.test()
            self.model.save(accuracy, epoch)

            # if self.top_accuracy < accuracy:
            #     self.top_accuracy = accuracy
            #     self.model.save(accuracy)

    def train(self):
        print('Training Neural Network Started!')

        self.data_handler.set_train_data()
        self.data_handler.shuffle()
        n_batch = int(self.data_handler.get_batch_count())

        for batch_index in tqdm(range(n_batch)):
            batch = self.data_handler.get_next()
            loss = self.model.train_batch(batch, batch_index)
            # print('Iterator:', iterInfo, 'Loss:', loss)

        print('Training Neural Network Ended!')

    def test(self):
        print('Testing Neural Network Started!')

        self.data_handler.set_test_data()
        n_correct = 0
        n_total = 0

        for _ in tqdm(range(int(self.data_handler.get_batch_count()))):
            # iterInfo = self.loader.getIteratorInfo()
            # print('Iterator:', iterInfo)
            batch = self.data_handler.get_next()
            recognized = self.model.infer_batch(batch)

            # print('Ground truth -> Recognized')
            for i in range(len(recognized)):
                is_correct = batch.labels[i] == recognized[i]
                # print('[OK]' if is_correct else '[ERR]', '"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
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
