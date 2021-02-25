import cv2
import numpy as np


def binarizar(img, factor):
    return ((img > factor)*255).astype('uint8')


img = cv2.cvtColor(cv2.imread('./images/tornillo_fondo_negro.jpg'),
                   cv2.COLOR_BGR2GRAY).astype('uint8')
img = binarizar(img, 100)

# Convertimos la imagen en 1D, sumando las columnas
horizontal = np.sum(img, axis=0)
# Convertimos la imagen en 1D, sumando las filas
vertical = np.sum(img, axis=1)

# Obtenemos los índices de los elementos no nulos
rango_horizontal = np.nonzero(horizontal)[0]
rango_vertical = np.nonzero(vertical)[0]

# El ancho y alto es la diferencia entre el primer y el último elemnto no nulo +1
ancho = rango_horizontal[-1]-rango_horizontal[0]+1
alto = rango_vertical[-1]-rango_vertical[0]+1

print(f'El elemento mide {ancho}x{alto} pixeles')
