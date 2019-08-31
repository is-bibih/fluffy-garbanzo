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

	def __str__(self):
		return('Nube: \nvolumen: {v}\nforma: {f}\nhumedad: {h}\nedad: {e}' \
			   .format(v=self.vol, f=self.forma, h=self.humedad, e=self.edad))
