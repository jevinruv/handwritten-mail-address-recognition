from tqdm import tqdm

from DataHandler import DataHandler
from Model import Model

path_dataset = "../../../../../../Dataset/"
# path_dataset = "../../../Dataset/"
path_test_img = '../resources/'
file_char_list = '../resources/chars.txt'
file_test_img = '../resources/test1.png'
n_epochs = 2


class Main:

    def __init__(self):

        # init class variables
        self.model = None
        self.loader = None
        self.top_accuracy = 0

    def create_new_model(self):

        # init data preparation
        self.loader = DataHandler(path_dataset, Model.batch_size, Model.img_size, Model.text_length)

        self.model = Model(self.loader.charList)

        # save characters of model for inference mode
        open(file_char_list, 'w').write(str().join(self.loader.charList))

        for epoch in range(n_epochs):
            print('Epoch ', epoch, ' of ', n_epochs)
            self.train()
            accuracy = self.test()
            if self.top_accuracy < accuracy:
                self.top_accuracy = accuracy
                self.model.save(accuracy)

    def train(self):
        print('Training Neural Network Started!')

        self.loader.set_train_data()
        self.loader.shuffle()
        n_batch = int(self.loader.get_batch_count())

        for batch_index in tqdm(range(n_batch)):
            # iterInfo = self.loader.getIteratorInfo()
            batch = self.loader.get_next()
            loss = self.model.train_batch(batch, batch_index)
            # print('Iterator:', iterInfo, 'Loss:', loss)

        print('Training Neural Network Ended!')

    def test(self):
        print('Testing Neural Network Started!')

        self.loader.set_test_data()
        n_correct = 0
        n_total = 0

        for _ in tqdm(range(int(self.loader.get_batch_count()))):
            # iterInfo = self.loader.getIteratorInfo()
            # print('Iterator:', iterInfo)
            batch = self.loader.get_next()
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

