import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt


def show_images(images, titles):
    for i in range(len(images)):
        plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray',)
        plt.title(titles[i])
        plt.axis(False)
    plt.show()


img = cv2.imread('./images/llave.jpg', 0)

# img = cv2.GaussianBlur(img, (3, 3), 0)  # Uncomment to use previous filter

# Sobel filter is equivalent to Canny with equal thresholds
sobel = cv2.Canny(img, 100, 100)
canny = cv2.Canny(img, 80, 600)

show_images([img, sobel, canny], ['Original', 'Sobel 100', 'Canny 40, 200'])
