import cv2
import numpy as np
from matplotlib import pyplot as plt


def transform(img, transform_matrix):
    transform_inverse = np.linalg.inv(transform_matrix)
    # Copia de la imagen en negro
    transformed_img = np.zeros(shape=img.shape)
    X = img.shape[0]
    Y = img.shape[1]
    for x in range(X):
        for y in range(Y):
            # Calculamos las coordenadas originales mediante la matriz inversa (como entero)
            new_pos = np.matmul(transform_inverse, [
                                x, y, 1]).astype('uint16')[:2]
            # Comprobamos que la posicion esta en rango de la matriz original
            if 0 <= new_pos[0] < X and 0 <= new_pos[1] < Y:
                # Asignamos a la nueva coordenada el valor de la anterior
                transformed_img[x, y] = img[new_pos[0], new_pos[1]]
    return transformed_img


transform_matrix = np.array([[.707, -.707, 0], [.707, .707, 0], [0, 0, 1]])

img = cv2.cvtColor(cv2.imread('./images/caja01.png'),
                   cv2.COLOR_BGR2GRAY).astype('uint8')

plt.subplot(1, 2, 1), plt.title('Original'), plt.imshow(
    img, 'gray'), plt.axis(False)
plt.subplot(1, 2, 2), plt.title('Editada'), plt.imshow(
    transform(img, transform_matrix), 'gray'), plt.axis(False)
plt.show()
