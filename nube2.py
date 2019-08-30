from nube import *
from math import pi

class Nube2(Nube):
	def __init__(self, vol, forma, humedad, edad=1, color='blanco', *args, **kwargs):
		super(Nube2, self).__init__()
		self.color = self.color
		self.vol += 10
		self.humedad = self.humedad**pi

	def cambiar_color(self):
		colores = ('rojo', 'naranja', 'amarillo', 'verde', 'azul', 'indigo', 'violeta')
		viejo = self.color
		i = len(self.forma) % 7
		self.color = colores[i]
		print('el color previo era {} y se cambió a {}'.format(viejo, self.color))


def prueba2(*args):
	color = input('de qué color es la nube: ')
	nube = Nube2(*args, color=color)
	nube.cambiar_color()

prueba2(*pedir_valores())
