from noiselib import add_gaussian_noise, add_salt_and_pepper_noise
import cv2
import numpy as np
from matplotlib import pyplot as plt


def min_max_filter(src, tam, filter):
    '''Devuelve la imagen dst resultante de pasar un filtro del minimo/maximo 
    de tamaño tam a la imagen src'''
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tam, tam))
    if filter == 'min':
        return cv2.erode(src, kernel)
    elif filter == 'max':
        return cv2.dilate(src, kernel)


def get_mean_filter(size):
    return np.ones((size, size)) / size**2


img = cv2.imread('./images/caja.png', cv2.IMREAD_GRAYSCALE)

img_gauss_noise = add_gaussian_noise(img, mean=0, sigma=20)
img_salt_noise = add_salt_and_pepper_noise(img, density=0.02)

# min max median gauss linear comparison 3x3
img_gauss_min = min_max_filter(img_gauss_noise, 3, 'min')
img_gauss_max = min_max_filter(img_gauss_noise, 3, 'max')
img_gauss_median = cv2.medianBlur(img_gauss_noise, 3)
img_gauss_gauss = cv2.GaussianBlur(img_gauss_noise, (3, 3), 0)
img_gauss_linear = cv2.filter2D(img_gauss_noise, -1, get_mean_filter(3))

plt.subplot(3, 2, 1), plt.title('Ruido medio, sigma 20'), plt.imshow(
    img_gauss_noise, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(3, 2, 2), plt.title('Suavizado minimo, 3x3'), plt.imshow(
    img_gauss_min, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(3, 2, 3), plt.title('Suavizado maximo, 3x3'), plt.imshow(
    img_gauss_max, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(3, 2, 4), plt.title('Suavizado mediana, 3x3'), plt.imshow(
    img_gauss_median, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(3, 2, 5), plt.title('Suavizado gaussiano, 3x3'), plt.imshow(
    img_gauss_gauss, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(3, 2, 6), plt.title('Suavizado lineal, 3x3'), plt.imshow(
    img_gauss_linear, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.show()

# min max and median comparison 5x5
img_gauss_min = min_max_filter(img_gauss_noise, 5, 'min')
img_gauss_max = min_max_filter(img_gauss_noise, 5, 'max')
img_gauss_median = cv2.medianBlur(img_gauss_noise, 5)
img_gauss_gauss = cv2.GaussianBlur(img_gauss_noise, (5, 5), 0)
img_gauss_linear = cv2.filter2D(img_gauss_noise, -1, get_mean_filter(5))

plt.subplot(3, 2, 1), plt.title('Ruido medio, sigma 20'), plt.imshow(
    img_gauss_noise, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(3, 2, 2), plt.title('Suavizado minimo, 5x5'), plt.imshow(
    img_gauss_min, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(3, 2, 3), plt.title('Suavizado maximo, 5x5'), plt.imshow(
    img_gauss_max, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(3, 2, 4), plt.title('Suavizado mediana, 5x5'), plt.imshow(
    img_gauss_median, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(3, 2, 5), plt.title('Suavizado gaussiano, 5x5'), plt.imshow(
    img_gauss_gauss, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(3, 2, 6), plt.title('Suavizado lineal, 5x5'), plt.imshow(
    img_gauss_linear, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.show()

img_salt_min = min_max_filter(img_salt_noise, 3, 'min')
img_salt_max = min_max_filter(img_salt_noise, 3, 'max')
img_salt_median = cv2.medianBlur(img_salt_noise, 3)
img_salt_gauss = cv2.GaussianBlur(img_salt_noise, (3, 3), 0)
img_salt_linear = cv2.filter2D(img_salt_noise, -1, get_mean_filter(3))

plt.subplot(3, 2, 1), plt.title('Ruido sal y pimienta, densidad 0.02'), plt.imshow(
    img_salt_noise, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(3, 2, 2), plt.title('Suavizado minimo, 3x3'), plt.imshow(
    img_salt_min, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(3, 2, 3), plt.title('Suavizado maximo, 3x3'), plt.imshow(
    img_salt_max, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(3, 2, 4), plt.title('Suavizado mediana, 3x3'), plt.imshow(
    img_salt_median, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(3, 2, 5), plt.title('Suavizado gaussiano, 3x3'), plt.imshow(
    img_salt_gauss, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(3, 2, 6), plt.title('Suavizado lineal, 3x3'), plt.imshow(
    img_salt_linear, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.show()
