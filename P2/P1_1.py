import cv2
import numpy as np
from matplotlib import pyplot as plt

Sx = 1
Sy = 2

transform_matrix = np.array([[Sx, 0, 0], [0, Sy, 0], [0, 0, 1]])
transform_inverse = np.linalg.inv(transform_matrix)

img = cv2.cvtColor(cv2.imread('./images/Estanteria.bmp'),
                   cv2.COLOR_BGR2GRAY).astype('uint8')

# Copia de la imagen en negro
transformed_img = np.zeros(shape=img.shape)
X = img.shape[0]
Y = img.shape[1]

# Iteramos en cada pixel
for x in range(X):
    for y in range(Y):
        # Calculamos las coordenadas transformadas (como entero)
        new_pos = np.matmul(transform_matrix, [x, y, 1]).astype('uint16')[:2]
        # Comprobamos que la nueva posicion esta en rango
        if new_pos[0] >= 0 and new_pos[1] >= 0 and new_pos[0] < X and new_pos[1] < Y:
            # Asignamos a la nueva coordenada el valor de la anterior
            transformed_img[new_pos[0], new_pos[1]] = img[x, y]

plt.subplot(1, 2, 1), plt.title('Original'), plt.imshow(
    img, 'gray'), plt.axis(False)
plt.subplot(1, 2, 2), plt.title('Editada'), plt.imshow(
    transformed_img, 'gray'), plt.axis(False)
plt.show()