import cv2 as cv2
from matplotlib import pyplot as plt

img = cv2.imread('./images/texto_sombras.bmp', 0)

_, bin_127 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)


def show_images(images, titles):
    for i in range(len(images)):
        plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.axis(False)
    plt.show()


# Binarización adaptativa con tamaño de bloque 21 y con 2 niveles debajo la media
bin_mean = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 2)
bin_gauss = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)

titles = ['Original', 'Threshold 127',
          'Adaptivo media, tamaño 21', 'Adaptivo Gaussiano, tamaño 21']
images = [img, bin_127, bin_mean, bin_gauss]
show_images(images, titles)

# Binarización adaptativa con tamaño de bloque 121 y con 2 niveles debajo la media
bin_mean = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 151, 2)
bin_gauss = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, 2)

titles = ['Original', 'Threshold 127',
          'Adaptivo media, tamaño 121', 'Adaptivo Gaussiano, tamaño 121']
images = [img, bin_127, bin_mean, bin_gauss]
show_images(images, titles)
