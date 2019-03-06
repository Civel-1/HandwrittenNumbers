import time

import numpy
import NeuralNetwork
from random import randint

# Classe permettant de faire le lien entre l'interface et le réseau de neurones
class NetworkHelper:

    def __init__(self, uinterface, hidden_nodes_value, learning_rate_value):
        # paramètres du réseau créé par défaut.
        str(learning_rate_value.replace(",", "."))
        self.interface = uinterface
        self.input_nodes = 784
        self.hidden_nodes = int(hidden_nodes_value)
        self.output_nodes = 10
        self.learning_rate = float(learning_rate_value)

        # crée une instance de base pour le réseau de neurones
        self.n = NeuralNetwork.NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes,
                                             self.learning_rate)
        # récupèration des informations depuis le fichier d'entraînement
        data_file = open("mnist_train.csv", 'r')
        self.data_list_train = data_file.readlines()
        self.data_train_length = self.data_list_train.__len__()
        data_file.close()
        # récupèration des informations depuis le fichier de test
        data_file = open("mnist_test.csv", 'r')
        self.data_list_test = data_file.readlines()
        self.data_test_length = self.data_list_test.__len__()
        data_file.close()

        print("Network created with " + str(self.hidden_nodes) + " hidden nodes and a " + str(
            self.learning_rate) + " learning rate")

    # Entraînement du réseau pour une epoch
    def train(self):

        self.interface.is_computing = True # valeur permettant à l'interface de savoir que le réseau est en phase d'entrainement

        compteur = 0
        self.interface.progress_value = 0 # valeur pour la progress bar de l'interface

        # Pour chaque nombre, on entraîne le réseau en lui donnant les données et le résultat attendu
        for i in range(self.data_train_length):
            all_values_to_train = self.data_list_train[i].split(',')
            # On normalise les valeurs en évitant l'existence de potentiel 0 qui posent problème dans le réseau.
            # On enlève la première valeur du vecteur qui correspond à la réponse
            scaled_train_input = (numpy.asfarray(all_values_to_train[1:]) / 255.0 * 0.99) + 0.01

            # Préparation du vecteur de résultat attendu, 0,01 pour tous et 0.99 pour le résultat attendu
            targets_outputs = numpy.zeros(self.output_nodes) + 0.01
            targets_outputs[int(all_values_to_train[0])] = 0.99
            self.n.train(scaled_train_input, targets_outputs)

            # mise à jour de la valeur de la progress bar de l'interface
            compteur += 1
            if compteur % 600 == 0:
                self.interface.progress_value += 1

            print(str(compteur) + " " + str(self.data_train_length))
        print("trained for one epoch")
        self.interface.is_computing = False

        return 0

    # Test la totalité des nombres présents dans le fichier de test et comptabilisation des résultats
    def test(self):
        total_answer = 0
        good_answer = 0

        # valeurs pour l'interface et l'avancement de la barre de progression
        self.interface.progress_value = 0
        self.interface.is_computing = True

        # on itère sur chacun des 10 000 nombres présents dans le fichier de test
        for i in range(self.data_test_length):
            all_values = self.data_list_test[i].split(',')
            # On normalise les valeurs en évitant l'existence de potentiel 0 qui posent problème dans le réseau.
            # On enlève la première valeur du vecteur qui correspond à la réponse
            scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # On questionne le réseau et récupération des résultats
            network_answer_list = self.n.query(scaled_input)

            targets = numpy.zeros(self.output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99

            # On récupère le résultat issu du réseau
            current_max_nn = 0
            current_nn_index = -1
            for index, y in enumerate(network_answer_list):
                if y > current_max_nn:
                    current_max_nn = y
                    current_nn_index = index

            # On récupère le résultat attendu issu des données
            current_max_answer = 0
            current_answer_index = -1
            for index, y in enumerate(targets):
                if y > current_max_answer:
                    current_max_answer = y
                    current_answer_index = index

            # On compare les résultats et met à jour le pourcentage de réussite
            if current_nn_index == current_answer_index:
                good_answer += 1
                pass
            total_answer += 1

            # Mise à jour de la progression
            if total_answer % 100 == 0:
                self.interface.progress_value += 1
            print("correct answer : " + str(current_answer_index) + "; guessed answer : " + str(current_nn_index))

        self.interface.is_computing = False
        return good_answer/total_answer

    # Renvoie les données d'un nombre aléatoire depuis les données du fichier de test
    def get_random_value(self):
        random = randint(0, self.data_test_length)
        all_values = self.data_list_test[random].split(',')

        return all_values

    # Test le réseau pour un nombre donnée en paramètres
    def guess_result(self, all_values):

        # normalisation et préparation des données
        scaled_input = (numpy.asfarray(all_values) / 255.0 * 0.99) + 0.01
        network_answer_list = self.n.query(scaled_input)
        return network_answer_list

