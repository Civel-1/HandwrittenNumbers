from functools import partial

import PIL.Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import NetworkHelper
from DrawNumberWindow import DrawNumberWindow
import numpy
from tkinter import ttk
from tkinter import *
from multiprocessing.pool import ThreadPool
from time import sleep
import matplotlib

matplotlib.use('TkAgg')

#Classe d'interface utilisant la librairire tkinter.
class UInterface:


    #Mise en place des éléments graphiques de l'interface
    def __init__(self):
        self.window = Tk()
        self.window.title("Reconnaissance de nombres")
        self.window.geometry("800x500")
        self.hidden_user_entry = int()
        self.lr_user_entry = float()
        self.is_computing = False
        self.drawing_values = None

        self.neural_label_frame = LabelFrame(self.window, text="Create new neural network with parameters")
        self.neural_label_frame.grid(sticky="WENS", row=0, column=0, padx=10, pady=10)
        self.entries_frame = Frame(self.neural_label_frame)
        self.entries_frame.grid(row=0, column=0)
        self.actions_label_frame = LabelFrame(self.window, text="Actions")
        self.actions_label_frame.grid(row=1, column=0, padx=10, pady=10, sticky="NESW")
        self.display_frame = Frame(self.window)
        self.display_frame.grid(row=0, column=1, rowspan=2, sticky="NESW")

        Label(self.entries_frame, text="Number of hidden nodes :", anchor="w").grid(sticky="WE", row=0, column=0, pady=(5, 0))
        self.hidden_nodes_entry = Entry(self.entries_frame, textvariable=self.hidden_user_entry)
        self.hidden_nodes_entry.insert(0, "100")
        self.hidden_nodes_entry.grid(row=1, column=0, padx=5, pady=5)
        self.hidden_nodes_entry.bind("<Button>", self.delete_placeholder_hidden)

        Label(self.entries_frame, text="Learning rate :", anchor="w").grid(sticky="WE", row=2, column=0, pady=(5, 0))
        self.learning_rate_entry = Entry(self.entries_frame, textvariable=self.lr_user_entry)
        self.learning_rate_entry.insert(0, "0.2")
        self.learning_rate_entry.grid(row=3, column=0, padx=5, pady=5)
        self.learning_rate_entry.bind("<Button>", self.delete_placeholder_learning)

        Label(self.entries_frame, text="Learning rate :", anchor="w").grid(sticky="WE", row=2, column=0, pady=(5, 0))
        Button(self.neural_label_frame, text="Create", command=partial(self.create_network)).grid(row=0, column=1, padx=(20, 0))

        Label(self.actions_label_frame, text="Train neural network :", anchor="w").grid(sticky="WE", row=0, column=0, pady=(5, 0))
        Button(self.actions_label_frame, text="Train for 1 epoch", command=partial(self.train)).grid(sticky="W", row=1, column=0, padx=10, pady=(0, 10))
        Label(self.actions_label_frame, text="Guess and display one random number :", anchor="w").grid(sticky="WE", row=2, column=0, pady=(5, 0))
        Button(self.actions_label_frame, text="Test one ", command=partial(self.test_one)).grid(sticky="W", row=3, column=0, padx=10, pady=(0, 10))
        Label(self.actions_label_frame, text="Guess over 10 000 numbers :", anchor="w").grid(sticky="WE", row=4, column=0, pady=(5, 0))
        self.test_frame = Frame(self.actions_label_frame)
        self.test_frame.grid(sticky="WE", row=5, column=0)
        Button(self.test_frame, text="Test", command=partial(self.test)).grid(sticky="WE", row=0, column=0, padx=10, pady=(0, 10))
        self.test_result = Label(self.test_frame, text="% of good answers ")
        self.test_result.grid(sticky="WNE", row=0, column=1)

        Label(self.actions_label_frame, text="Draw your number :", anchor="w").grid(sticky="WE", row=6, column=0, pady=(5, 0))
        Button(self.actions_label_frame, text="Draw ", command=partial(self.draw)).grid(sticky="W", row=8, column=0, padx=10, pady=(0, 10))

        self.progress = ttk.Progressbar(orient="horizontal", maximum=100, mode="determinate")
        self.progress_value = 0

        # Création du réseau de neurones utilisé par défaut
        self.nh = NetworkHelper.NetworkHelper(self, self.hidden_nodes_entry.get(), self.learning_rate_entry.get())

        self.window.mainloop()

    # Méthode du bouton "Create new neural network"
    # Recrée un nouveau réseau avec les paramètres souhaités par l'utilisateur
    def create_network(self):
        self.nh = NetworkHelper.NetworkHelper(self, self.hidden_nodes_entry.get(), self.learning_rate_entry.get())

    # Méthode d'affichage des résultats à travers deux graphs en utilisant la librairie matplotlib
    def display_result(self, all_values, results, is_drawing):

        numbers = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        x_pos = [0, 1]
        y_pos = numpy.arange(len(numbers))
        # Petite différence selon la source de l'appel. Si dessin, pas de label (et donc vecteur de 784 valeurs).
        if is_drawing:
            image_array = numpy.asfarray(all_values).reshape((28, 28))
        else:
            image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
        fig = Figure(figsize=(2, 2))
        a = fig.add_subplot(111)
        a.imshow(image_array, cmap='Greys', interpolation='None')
        a.axis('off')
        fig2 = Figure(figsize=(5, 2))
        b = fig2.add_subplot(211)
        rect = b.bar(y_pos, results, align='center', alpha=0.5)
        b.spines['right'].set_visible(False)
        b.spines['top'].set_visible(False)
        b.axes.set_ylim([0, 1])
        b.patch.set_alpha(1)
        b.axes.set_yticks(x_pos, minor=False)
        b.set_yticklabels(['0', '1'])
        b.axes.set_xticks(y_pos)
        b.set_xticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

        # Ajout des valeurs des probabilités en haut des barres
        for r in rect:
            height = r.get_height()
            b.text(r.get_x() + r.get_width() / 2., 1.05 * height, '%.2f' % float(height),
                   ha='center', va='bottom')

        canvas = FigureCanvasTkAgg(fig, master=self.display_frame)
        canvas2 = FigureCanvasTkAgg(fig2, master=self.display_frame)

        Label(self.display_frame, text="Input image :").grid(sticky="WN", row=0, column=0, pady=(15, 0))
        canvas.get_tk_widget().grid(sticky="WN", row=1, column=0)
        canvas.draw()
        Label(self.display_frame, text="Results :").grid(sticky="WN", row=2, column=0, pady=(30, 0))
        canvas2.get_tk_widget().grid(sticky="WN", row=3, column=0)
        canvas2.draw()

    # Méthode répondant au bouton "Test one"
    # Prend aléatoirement un nombre issu du fichier de test et questionne le réseau. Appelle ensuite la fonction
    # d'affichage des résultats
    def test_one(self):
        all_values = self.nh.get_random_value()
        results = self.nh.guess_result(all_values[1:])
        print(all_values)
        self.display_result(all_values, results, False)

    # Méthode répondant au bouton "Test"
    # Teste la totalité des nombres présents dans le fichier de test et affiche le pourventage de réussite
    def test(self):
        # Multithreading pour ne pas gelé l'application durant le traitement.
        pool = ThreadPool(processes=1)
        self.progress.grid(row=3, column=1, sticky="W")
        async_result = pool.apply_async(self.nh.test)
        self.progress["value"] = 0
        self.window.update()

        # Mise à jour de la progress bar durant le travail de fond. self.progress_value est mis à jour par le networkHelper
        while self.is_computing:
            sleep(0.25)
            self.progress["value"] = self.progress_value
            self.window.update()
        results = async_result.get()

        self.test_result.config(text=str(results) + "% of good answers")
        self.progress.grid_forget()


    # Méthode répondant au bouton "Train for one epoch"
    # Apprentissage du réseau grâce aux nombres présents dans le fichier d'entraînement
    def train(self):
        # Multithreading
        pool = ThreadPool(processes=1)
        self.progress.grid(row=3, column=1, sticky="W")
        async_result = pool.apply_async(self.nh.train)
        self.progress["value"] = 0
        self.window.update()

        # Mise à jour de la progress bar durant le travail de fond. self.progress_value est mis à jour par le networkHelper
        while self.is_computing:
            sleep(0.25)
            self.progress["value"] = self.progress_value
            self.window.update()

        async_result.get()
        self.progress.grid_forget()

    # Méthode répondant au bouton "Draw"
    # Ouvre une fenêtre pour dessiner à la souris un nombre
    # --------- NE FONCTIONNE PAS ----------------
    def draw(self):
        DrawNumberWindow(self)

    # Méthode répondant au bouton "Train for one epoch"
    # Apprentissage du réseau grâce aux nombres présents dans le fichier d'entraînement
    def guess_drawing(self):

        # From image .png issue de gimp
        # image = PIL.Image.open("image.png", "r").resize((28, 28)).convert("L")
        array = 255 - self.drawing_values.reshape(784)

        results = self.nh.guess_result(array)
        self.display_result(array, results, True)

    # Méthode qui permet d'effacer les paramètres par défauts lorsque l'utilisateur clique pour les modifier
    def delete_placeholder_hidden(self, event):
        self.hidden_nodes_entry.delete(0, END)

    # Méthode qui permet d'effacer les paramètres par défauts lorsque l'utilisateur clique pour les modifier
    def delete_placeholder_learning(self, event):
        self.learning_rate_entry.delete(0, END)
