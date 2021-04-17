# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 21:03:47 2021
Funciones para trabajar con conjuntos de puntos en RANSAC
@author: eusebio
"""

import random
import numpy as np
import cv2

def get_random_points( puntos,N):
    '''Elige N puntos aleatoriamente del conjunto (xPixelsCont,yPixelsCont)
    Devuelve dos vectores: xSample con las coord x y ySample con las coord y'''
    samples=[]

    numPixelsCont=len(puntos)
    #N numero de puntos aleatorios
    for k in range(N):
        indexRand=random.randrange(0,numPixelsCont)
        samples.append( puntos[indexRand] )
    return np.array(samples)


def get_contour_points(img):
    '''img es una imagen binaria solo con los puntos de contorno'''
    y,x=np.where(img)
    cnt=[]
    for i in range(len(x)):
        cnt.append([[x[i],y[i]]])
        array_ptos_cnt=np.array(cnt)
    return array_ptos_cnt