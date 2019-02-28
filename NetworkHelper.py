import numpy
import NeuralNetwork
from random import randint


class NetworkHelper:

    def __init__(self, uinterface, hidden_nodes_value, learning_rate_value):
        # number of input, hidden and output nodes
        str(learning_rate_value.replace(",", "."))
        self.interface = uinterface
        self.input_nodes = 784
        self.hidden_nodes = int(hidden_nodes_value)
        self.output_nodes = 10
        self.learning_rate = float(learning_rate_value)
        # create instance of neural network
        self.n = NeuralNetwork.NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes,
                                             self.learning_rate)
        data_file = open("mnist_train.csv", 'r')
        self.data_list_train = data_file.readlines()
        self.data_train_length = self.data_list_train.__len__()
        data_file.close()
        data_file = open("mnist_test.csv", 'r')
        self.data_list_test = data_file.readlines()
        self.data_test_length = self.data_list_test.__len__()
        data_file.close()
        print("Network created with " + str(self.hidden_nodes) + " hidden nodes and a " + str(
            self.learning_rate) + " learning rate")

    def train(self):
        # preparing data and training
        self.interface.is_computing = True
        compteur = 0
        self.interface.progress_value = 0
        for i in range(self.data_train_length):
            all_values_to_train = self.data_list_train[i].split(',')
            scaled_train_input = (numpy.asfarray(all_values_to_train[1:]) / 255.0 * 0.99) + 0.01

            targets_outputs = numpy.zeros(self.output_nodes) + 0.01
            targets_outputs[int(all_values_to_train[0])] = 0.99
            self.n.train(scaled_train_input, targets_outputs)
            compteur += 1
            if compteur % 600 == 0:
                self.interface.progress_value += 1

            print(str(compteur) + " " + str(self.data_train_length))
        print("trained for one epoch")
        self.interface.is_computing = False
        return 0

    def test(self):
        total_answer = 0
        good_answer = 0
        self.interface.progress_value = 0
        self.interface.is_computing = True
        # preparing data and testing
        for i in range(self.data_test_length):
            all_values = self.data_list_test[i].split(',')
            scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            network_answer_list = self.n.query(scaled_input)

            targets = numpy.zeros(self.output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99

            current_max_nn = 0
            current_nn_index = -1
            for index, y in enumerate(network_answer_list):
                if y > current_max_nn:
                    current_max_nn = y
                    current_nn_index = index

            current_max_answer = 0
            current_answer_index = -1
            for index, y in enumerate(targets):
                if y > current_max_answer:
                    current_max_answer = y
                    current_answer_index = index

            if current_nn_index == current_answer_index:
                good_answer += 1
                pass
            total_answer += 1
            if total_answer % 100 == 0:
                self.interface.progress_value += 1
            print("correct answer : " + str(current_answer_index) + "; guessed answer : " + str(current_nn_index))

        self.interface.is_computing = False
        return good_answer/total_answer

    def get_random_value(self):
        random = randint(0, self.data_test_length)

        all_values = self.data_list_test[random].split(',')

        return all_values

    def guess_result(self, all_values):

        scaled_input = (numpy.asfarray(all_values) / 255.0 * 0.99) + 0.01
        network_answer_list = self.n.query(scaled_input)
        return network_answer_list

