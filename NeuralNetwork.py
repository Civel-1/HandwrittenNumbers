import scipy.special
import numpy


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes,
                 learning_rate):
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        # learning rate
        self.lr = learning_rate
        self.activation_function = lambda x: scipy.special.expit(x)

        self.wih = numpy.random.rand(self.h_nodes, self.i_nodes) - 0.5
        self.who = numpy.random.rand(self.o_nodes, self.h_nodes) - 0.5

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list).T
        targets = numpy.array(targets_list).T
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        temp = 1.0 - final_outputs
        self.who += self.lr * numpy.dot((output_errors * final_outputs * temp)[:, None],
                                        numpy.transpose(hidden_outputs)[None, :])
        # update the weights for the links between the input and hidden layers
        temp2 = 1.0 - hidden_outputs
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs *
                                         temp2)[:, None], numpy.transpose(inputs)[None, :])

    pass

    # query the neural network
    def query(self, inputs_list):
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs_list)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
