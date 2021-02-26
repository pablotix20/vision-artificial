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


img = cv2.cvtColor(cv2.imread('./images/caja06.png'),
                   cv2.COLOR_BGR2GRAY).astype('uint8')

m1 = np.array([[1, 0, 100], [0, 1, 60], [0, 0, 1]])
m2 = np.array([[.707, -.707, 0], [.707, .707, 0], [0, 0, 1]])
m3 = np.array([[1, 0, -100], [0, 1, -60], [0, 0, 1]])

img1 = transform(img, m1)
img2 = transform(img1, m2)
img3 = transform(img2, m3)

plt.subplot(2, 2, 1), plt.title('Original'), plt.imshow(
    img, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 2), plt.title('Editada 1'), plt.imshow(
    img1, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 3), plt.title('Editada 2'), plt.imshow(
    img2, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 4), plt.title('Editada 3'), plt.imshow(
    img3, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.show()
