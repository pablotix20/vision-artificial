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


cojinetes = cv2.imread('./images/cojinetes.bmp', 0)

cojinetes_noisy = add_gaussian_noise(cojinetes, sigma=10)

laplacian = cv2.Laplacian(cojinetes, cv2.CV_64F)
laplacian_noisy = cv2.Laplacian(cojinetes_noisy, cv2.CV_64F)

show_images([cojinetes, laplacian, cojinetes_noisy, laplacian_noisy],
            ['Original', 'Laplaciana', 'Con ruido sigma 10', 'Laplaciana con ruido'])

show_images([cojinetes, laplacian**2, cojinetes_noisy, laplacian_noisy**2],
            ['Original', 'Laplaciana', 'Con ruido sigma 10', 'Laplaciana con ruido'])

gauss = cv2.GaussianBlur(cojinetes_noisy, (3, 3), cv2.CV_64F)
diff = cojinetes_noisy.astype(np.float64)-gauss.astype(np.float64)

show_images([cojinetes_noisy, gauss, diff, (diff**2)**.5], [
            'Original', 'Filtrada', 'Diferencia', 'abs(diferencia)'])
