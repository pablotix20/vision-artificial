import cv2
from matplotlib import pyplot as plt
import numpy as np


def editar_histograma(img, m):
    edit = np.array(img)-m
    edit[edit > img] = 0  # Solucionamos overflow
    m = np.amax(edit)  # Obtenemos el valor máximo
    edit = edit*(255/m)  # Escalamos hasta el valor maximo
    return edit.astype('uint8')


def calc_hist(img):
    filas, columnas = img.shape[:2]
    histo = np.zeros(256, dtype=int)

    for i in range(filas):
        for j in range(columnas):
            histo[img[i][j]] += 1

    return(histo)


img = cv2.cvtColor(cv2.imread('./images/deportivo.jpg'),
                   cv2.COLOR_BGR2GRAY).astype('uint8')
m = int(input('Factor de reduccion m: '))
img_edit = editar_histograma(img, m)

hist = calc_hist(img)
hist_edit = calc_hist(img_edit)
x = np.linspace(0, 255, 256)

plt.subplot(2, 2, 1), plt.title('Original'), plt.imshow(
    img, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 2), plt.title('Editada'), plt.imshow(
    img_edit, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 3), plt.plot(x, hist)
plt.subplot(2, 2, 4), plt.plot(x, hist_edit)

plt.show()
