import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt

def show_images(images, titles):
    for i in range(len(images)):
        plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.axis(False)
    plt.show()

img = cv2.imread('./images/pcb.jpg', 0)

sobely=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
sobelx=np.transpose(sobely)
#Filtramos las imágenes con esas máscaras
dx= cv2.filter2D(img, cv2.CV_64F, sobelx) #en CV_64F. habrá valores positivos y negativos
dy= cv2.filter2D(img, cv2.CV_64F, sobely)

show_images([dx,dy],['dx','dy'])