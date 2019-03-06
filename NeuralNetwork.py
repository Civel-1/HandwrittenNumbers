import scipy.special
import numpy

# classe du réseau de neurone en lui même
class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes,
                 learning_rate):
        #paramètres du réseau
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        self.lr = learning_rate
        self.activation_function = lambda x: scipy.special.expit(x)

        # Création des matrices de poids avec des valeurs aléatoires. - 0,5 pour que les valeurs soient entre -0,5 et 0,5
        self.wih = numpy.random.rand(self.h_nodes, self.i_nodes) - 0.5
        self.who = numpy.random.rand(self.o_nodes, self.h_nodes) - 0.5

    # Entraînement du réseau pour un nombre
    def train(self, inputs_list, targets_list):
        # ------ FEED FORWARD ----------
        # On convertit notre matrice d'input 28*28 en une liste de 784
        inputs = numpy.array(inputs_list).T
        targets = numpy.array(targets_list).T
        # propagation du signal dans la couche cachée
        hidden_inputs = numpy.dot(self.wih, inputs)
        # Calcul des valeurs issues de la couche cachée
        hidden_outputs = self.activation_function(hidden_inputs)
        # propagation du signal dans la couche de sortie
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Calcul des valeurs issues de la couche de sortie (résultats finaux)
        final_outputs = self.activation_function(final_inputs)

        # ----- BACK PROPAGATION ------
        #Calcul des erreurs au niveau de la couche de sortie et de la couche cachée
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # Mise à jour des poids des liens entre la couche de sortie et la couche cachée
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs))[:, None],
                                        numpy.transpose(hidden_outputs)[None, :])
        # Mise à jour des poids des liens entre la couche cachée et la couche d'entrées
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs *
                                         (1.0 - hidden_outputs))[:, None], numpy.transpose(inputs)[None, :])

    pass

    # Questionnement du réseau de neurones pour un nombre donné en paramètres
    def query(self, inputs_list):
        # ------ FEED FORWARD ----------
        # propagation du signal dans la couche cachée
        hidden_inputs = numpy.dot(self.wih, inputs_list)
        # Calcul des valeurs issues de la couche cachée
        hidden_outputs = self.activation_function(hidden_inputs)
        # propagation du signal dans la couche de sortie
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Calcul des valeurs issues de la couche de sortie (résultats finaux)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
