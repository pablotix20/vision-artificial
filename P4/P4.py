import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt


def show_images(images, titles):
    for i in range(len(images)):
        plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray',)
        plt.title(titles[i])
        plt.axis(False)
    plt.show()


img = cv2.imread('./images/llanta.jpg', 0)

sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
sobelx = np.transpose(sobely)
# Filtramos las imágenes con esas máscaras
dx = cv2.filter2D(img, cv2.CV_64F, sobelx)
dy = cv2.filter2D(img, cv2.CV_64F, sobely)
d = (dx**2+dy**2)**.5

# Sobel filter is equivalent to Canny with equal thresholds
edges = cv2.Canny(img, 120, 120)

show_images([img, dx, dy, d], ['Original', 'dx', 'dy', 'Módulo gradiente'])
show_images([img, edges], ['Original', 'Bordes Sobel'])
