import cv2
from matplotlib import pyplot as plt
import numpy as np

def editar_tono(img):  
    R =  img[:,:,0]
    G =  img[:,:,1]
    R_bright = (R*1.5).astype('uint8')
    G_bright = (G*1.5).astype('uint8')
    R_bright[R_bright<R]=255 # Solucionar overflow uint8
    G_bright[G_bright<G]=255 # Solucionar overflow uint8
    red = np.array(img)
    red[:,:,0]=R_bright
    green = np.array(img)
    green[:,:,1]=G_bright

    plt.subplot(1,3,1), plt.title('Original'), plt.imshow(img), plt.axis(False)
    plt.subplot(1,3,2), plt.title('Enrojecida'), plt.imshow(red), plt.axis(False)
    plt.subplot(1,3,3), plt.title('Verdosa'), plt.imshow(green), plt.axis(False)
    plt.show()

def planos_grises(img):
    grayscale = (.3*img[:,:,2]+.59*img[:,:,1]+.11*img[:,:,0]).astype('uint8')
    inverted = 255-grayscale
    binary = grayscale>127
    bright = grayscale+100
    bright[bright<grayscale]=255 # Solucionar overflow uint8
    dark = grayscale-100
    dark[dark>grayscale]=0 # Solucionar overflow uint8
    colored = np.zeros(shape=(grayscale.shape[0],grayscale.shape[1],3)).astype('uint8')
    colored[:,:,1] = grayscale

    
    plt.subplot(2,3,1), plt.title('gris'), plt.imshow(grayscale,'gray'), plt.axis(False) 
    plt.subplot(2,3,2), plt.title('invertida'), plt.imshow(inverted,'gray'), plt.axis(False) 
    plt.subplot(2,3,3), plt.title('binarizada'), plt.imshow(binary,'gray'), plt.axis(False) 
    plt.subplot(2,3,4), plt.title('brilllante'), plt.imshow(bright,'gray'), plt.axis(False) 
    plt.subplot(2,3,5), plt.title('oscura'), plt.imshow(dark,'gray'), plt.axis(False) 
    plt.subplot(2,3,6), plt.title('coloreada'), plt.imshow(colored), plt.axis(False)    
    plt.show()

img = cv2.cvtColor(cv2.imread('./images/toysflash.png'), cv2.COLOR_BGR2RGB)

editar_tono(img)
# planos_grises(img)