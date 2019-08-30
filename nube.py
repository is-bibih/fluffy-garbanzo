class Nube(object):
	def __init__(self, vol, forma, humedad, edad=1, *args, **kwargs):
		self.vol = vol
		self.forma = forma
		self.humedad = humedad
		self.edad = edad

	def llover(self):
		if self.vol > 10 and self.humedad > 0.5:
			print('está lloviendo')
			self.vol -= 1
			self.humedad -= 0.1
		elif len(self.forma) < 6:
			print('podría llover después')
			self.vol += 1
			self.humedad += self.humedad*0.7
		else:
			print('nunca va a llover de esta nube')

	def disipar(self):
		if self.edad >= 5:
			print('se disipó la nube')
		else:
			print('la nube sigue ahí')
			self.edad += 1


def pedir_valores():
	print('nada tiene validación entonces por favor pon cosas razonables')
	vol = float(input('volumen de la nube (m^3): '))
	forma = input('cómo describirías la forma de la nube: ')
	humedad = float(input('proporción de la nube que es agua: '))
	edad = int(input('cuántos días tiene la nube: '))

	return (vol, forma, humedad, edad)

def prueba(vol, forma, humedad, edad):
	if not edad:
		nube = Nube(vol, forma, humedad)
	else:
		nube = Nube(vol, forma, humedad, edad)

	nube.llover()
	nube.disipar()

prueba(*pedir_valores())