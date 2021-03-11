from noiselib import add_gaussian_noise, add_salt_and_pepper_noise
import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_mean_filter(size):
    return np.ones((size, size)) / size**2


img = cv2.imread('./images/caja.png', cv2.IMREAD_GRAYSCALE)

img_gauss_noise = add_gaussian_noise(img, mean=0, sigma=20)
img_salt_noise = add_salt_and_pepper_noise(img, density=0.01)

img_filter_linear_3 = cv2.filter2D(img_gauss_noise, -1, get_mean_filter(3))
img_filter_linear_5 = cv2.filter2D(img_gauss_noise, -1, get_mean_filter(5))
img_filter_linear_7 = cv2.filter2D(img_gauss_noise, -1, get_mean_filter(7))

plt.subplot(2, 2, 1), plt.title('Ruido medio, sigma 20'), plt.imshow(
    img_gauss_noise, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 2), plt.title('Suavizado lineal, 3x3'), plt.imshow(
    img_filter_linear_3, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 3), plt.title('Suavizado lineal, 5x5'), plt.imshow(
    img_filter_linear_5, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 4), plt.title('Suavizado lineal, 7x7'), plt.imshow(
    img_filter_linear_7, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.show()

gauss_3x1 = cv2.getGaussianKernel(3, -1)
gauss_1x3 = np.transpose(gauss_3x1)
gauss_3x3 = gauss_3x1 @ gauss_1x3

gauss_5x1 = cv2.getGaussianKernel(5, -1)

img_filter_gauss_3 = cv2.filter2D(img_gauss_noise, -1, gauss_3x3)
img_filter_gauss_5 = cv2.sepFilter2D(
    img_gauss_noise, -1, gauss_5x1, gauss_5x1)
img_filter_gauss_7 = cv2.GaussianBlur(img_gauss_noise, (7, 7), 0)

plt.subplot(2, 2, 1), plt.title('Ruido medio, sigma 20'), plt.imshow(
    img_gauss_noise, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 2), plt.title('Suavizado gaussiano, 3x3'), plt.imshow(
    img_filter_gauss_3, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 3), plt.title('Suavizado gaussiano, 5x5'), plt.imshow(
    img_filter_gauss_5, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 4), plt.title('Suavizado gaussiano, 7x7'), plt.imshow(
    img_filter_gauss_7, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.show()

plt.subplot(2, 2, 1), plt.title('Suavizado gaussiano, 3x3'), plt.imshow(
    img_filter_gauss_3, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 2), plt.title('Suavizado gaussiano, 7x7'), plt.imshow(
    img_filter_gauss_7, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 3), plt.title('Suavizado lineal, 3x3'), plt.imshow(
    img_filter_linear_3, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 4), plt.title('Suavizado lineal, 7x7'), plt.imshow(
    img_filter_linear_7, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.show()

img_filter_salt_gauss = cv2.GaussianBlur(img_salt_noise, (3, 3), 0)
img_filter_salt_linear = cv2.filter2D(img_salt_noise, -1, get_mean_filter(3))

plt.subplot(2, 2, 1), plt.title('Ruido sal y pimienta'), plt.imshow(
    img_salt_noise, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 2), plt.title('Suavizado gaussiano, 3x3'), plt.imshow(
    img_filter_salt_gauss, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 3), plt.title('Suavizado lineal, 3x3'), plt.imshow(
    img_filter_salt_linear, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.show()
