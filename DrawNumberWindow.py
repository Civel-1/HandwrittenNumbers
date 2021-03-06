from PIL import Image, ImageDraw
import PIL
from tkinter import *
import numpy

# Nouvelle fenêtre permettant de dessiner un nombre à la souris
# NE FONCTIONNE PAS
class DrawNumberWindow:

    def __init__(self, interface):
        self.width = 224
        self.height = 224
        self.interface = interface
        self.second_window = Tk()
        self.cv = Canvas(self.second_window, width=self.width, height=self.height, bg='white')
        self.cv.grid(row=0, column=0)

        self.image1 = PIL.Image.new("L", (self.width, self.height))
        self.draw = ImageDraw.Draw(self.image1)
        self.cv.bind("<B1-Motion>", self.paint)

        button = Button(self.second_window, text="Guess", command=self.guess)
        button.grid(row=1, column=0)
        self.second_window.mainloop()

    # renvoie un vecteur de 784 valeurs correspondant au dessin recadré de l'utilisateur.
    def guess(self):
        image = self.image1.resize((28, 28))
        array = numpy.array(image.getdata(),
                    numpy.uint8).reshape(image.size[1], image.size[0], 1)
        self.interface.drawing_values = array
        self.interface.guess_drawing()
        self.second_window.destroy()

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.cv.create_oval(x1, y1, x2, y2, fill="black", width=3)
        self.draw.line([x1, y1, x2, y2], fill="black", width=3)
