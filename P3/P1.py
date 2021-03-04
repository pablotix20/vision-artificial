from noiselib import add_gaussian_noise, add_salt_and_pepper_noise, add_speckle_noise
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('./images/caja.png', cv2.IMREAD_GRAYSCALE)

img_gauss_std = add_gaussian_noise(img, mean=0, sigma=10)
img_gauss_bright = add_gaussian_noise(img, mean=25, sigma=10)
img_gauss_high = add_gaussian_noise(img, mean=0, sigma=50)

plt.subplot(2, 2, 1), plt.title('Original'), plt.imshow(
    img, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 2), plt.title('Ruido leve, sigma 10'), plt.imshow(
    img_gauss_std, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 3), plt.title('Ruido fuerte, sigma 50'), plt.imshow(
    img_gauss_high, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 4), plt.title('Ruido con media > 0'), plt.imshow(
    img_gauss_bright, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.show()

img_salt_std = add_salt_and_pepper_noise(img, density=0.01)
img_salt_med = add_salt_and_pepper_noise(img, density=0.05)
img_salt_high = add_salt_and_pepper_noise(img, density=0.25)

plt.subplot(2, 2, 1), plt.title('Original'), plt.imshow(
    img, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 2), plt.title('Ruido leve, densidad 0,01'), plt.imshow(
    img_salt_std, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 3), plt.title('Ruido medio, densidad 0,05'), plt.imshow(
    img_salt_med, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 4), plt.title('Ruido fuerte, densidad 0,25'), plt.imshow(
    img_salt_high, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.show()

img = cv2.imread('./images/llave.png', cv2.IMREAD_GRAYSCALE)
img_speckle_low = add_speckle_noise(img, sigma=0.1)
img_speckle_med = add_speckle_noise(img, sigma=0.5)
img_speckle_high = add_speckle_noise(img, sigma=1)

plt.subplot(2, 2, 1), plt.title('Original'), plt.imshow(
    img, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 2), plt.title('Ruido leve, sigma 0,1'), plt.imshow(
    img_speckle_low, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 3), plt.title('Ruido medio, sigma 0,5'), plt.imshow(
    img_speckle_med, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 4), plt.title('Ruido fuerte, sigma 1'), plt.imshow(
    img_speckle_high, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.show()
