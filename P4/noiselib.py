# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 19:01:37 2021
FUNCIONES PARA AÑADIR RUIDO A IMAGENES. GAUSSIANO, SPECKLE Y SAL Y PIMIENTA
@author: eusebio
"""

import numpy as np

def add_gaussian_noise(img, mean=0, sigma=1):
    '''añade a la imagen ruido gaussiano de media mean y desv.standar sigma'''
    row, col = img.shape
    gauss = np.array(np.random.normal(mean, sigma, (row, col)))
    noisy_image = np.clip((img + gauss), 0, 255)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image

def add_salt_and_pepper_noise(img, density=0.01):
    '''añade a la imagen ruido sal y pimienta a la imagen'''
    row, col = img.shape
    thres = 1 - density
    noisy_image = img.copy()
    for i in range(row):
        for j in range(col):
            prob = np.random.rand()
            if density < prob < thres:
                pass
            elif prob < density:
                noisy_image.itemset(i, j, 0)
            elif prob > thres:
                noisy_image.itemset(i, j, 255)
    return noisy_image

def add_speckle_noise(img, sigma=0.1):
    '''añade a la imagen ruido speckle (multiplicativo)'''
    row, col = img.shape
    gauss = np.array(np.random.normal(0, sigma, (row, col)))
    noisy_image = np.clip((img + img*gauss), 0, 255)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image
