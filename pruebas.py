import nube
import nube2

def pedir_valores():
	print('nada tiene validación entonces por favor pon cosas razonables')
	vol = float(input('volumen de la nube (m^3): '))
	forma = input('cómo describirías la forma de la nube: ')
	humedad = float(input('proporción de la nube que es agua: '))
	edad = int(input('cuántos días tiene la nube: '))

	return (vol, forma, humedad, edad)

def prueba(vol, forma, humedad, edad):
	if not edad:
		nub = nube.Nube(vol, forma, humedad)
	else:
		nub = nube.Nube(vol, forma, humedad, edad)
	nub.llover()
	nub.disipar()
	print(nub)

def prueba2(*args):
	args = [*args]
	color = input('de qué color es la nube: ')
	args.append(color)
	nub = nube2.Nube2(*args)
	nub.cambiar_color()

prueba2(*pedir_valores())
