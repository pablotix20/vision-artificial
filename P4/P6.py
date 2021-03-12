import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt
from noiselib import add_gaussian_noise


def show_images(images, titles):
    for i in range(len(images)):
        plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray',)
        plt.title(titles[i])
        plt.axis(False)
    plt.show()


llave = cv2.imread('./images/llave.jpg', 0)
cojinetes = cv2.imread('./images/cojinetes.bmp', 0)

llave = add_gaussian_noise(llave, sigma=20)
cojinetes = add_gaussian_noise(cojinetes, sigma=20)

img = llave
# img = cojinetes # Uncomment to change image

# Adjust aperture size
show_images([img, cv2.Canny(img, 200, 200, apertureSize=3), cv2.Canny(img, 2000, 2000,
                                                                      apertureSize=5), cv2.Canny(img, 40000, 40000, apertureSize=7)], ['Original', '3x3', '5x5', '7x7'])

show_images([img, cv2.Canny(img, 30000, 30000, apertureSize=7), cv2.Canny(img, 50000, 50000,
                                                                          apertureSize=7), cv2.Canny(img, 70000, 70000, apertureSize=7)], ['Original', '30k', '50k', '70k'])

show_images([img, cv2.Canny(img, 30000, 50000, apertureSize=7)],
            ['Original', '30k, 50k'])
