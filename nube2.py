from nube import *
from math import pi

class Nube2(Nube):
	def __init__(self, vol, forma, humedad, edad=1, color='blanco', *args, **kwargs):
		super(Nube2, self).__init__(vol, forma, humedad, edad)
		self.color = color
		print(self)
		self.vol += 10
		self.humedad = self.humedad**pi

	def cambiar_color(self):
		colores = ('rojo', 'naranja', 'amarillo', 'verde', 'azul', 'indigo', 'violeta')
		viejo = self.color
		i = len(self.forma) % 7
		self.color = colores[i]
		print('el color previo era {} y se cambió a {}'.format(viejo, self.color))

	def __str__(self):
		string = super(Nube2, self).__str__()
		string += '\ncolor: {}'.format(self.color)
		return string
