# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 19:57:48 2021
FUNCIONES PARA DIBUJAR: PUNTOS SEGMENTOS, RECTAS, ...
@author: eusebio
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

def points(img, points, color, grosor=3):
    '''Dibuja los puntos'''
    if len(img.shape)==2:#si img es de niveles de gris
        img_color=cv2.merge([img,img,img])
    else:
        img_color=img.copy()
    for i in range(len(points)):
        cv2.circle(img_color, (points[i][0][0],points[i][0][1]), grosor, color, -1)#centro,radio,color,macizo=-1
    plt.imshow(img_color)
    return img_color    
    

def segment_PQ(img, puntoP, puntoQ, color, grosor=3):
    '''Dibuja el segmento de P(x,y) a Q(x,y)'''
    if len(img.shape)==2:#si img es de niveles de gris
        img_color=cv2.merge([img,img,img])
    else:
        img_color=img.copy()
    cv2.line(img_color, puntoP, puntoQ, color, grosor , cv2.LINE_AA)
    plt.imshow(img_color)
    return img_color

def rect_PQ(img, puntoP, puntoQ, color, grosor=3): 
    if puntoQ[0]==puntoP[0]:#recta vertical
        #punto inicial
        P= (puntoP[0], 0)
        # #punto final
        Q= (puntoQ[0], img.shape[0]) 
        img_color=segment_PQ(img, P, Q, color)
    else:  #y=mx+b 
        m=(puntoQ[1]-puntoP[1])/(puntoQ[0]-puntoP[0])
        b= puntoQ[1] - m*puntoQ[0]
        P=(0,np.round(b))
        Q=(img.shape[1],np.round(m*img.shape[1]+b))
        img_color=segment_PQ(img, P, Q, color)
    plt.imshow(img_color)
    return img_color
    
    
def rect_ABC(img, rectaABC, color):
    '''Dibuja recta Ax + By + C=0.  Eje X es horizontal'''   
    [A,B,C]=rectaABC
    if B==0: #Recta vertical
        #punto inicial
        P= (int(np.round(-C/A)), 0)
        # #punto final
        Q= (int(np.round(-C/A)), img.shape[0]) 
        img_color=segment_PQ(img, P, Q, color)
    else:
        m=-A/B
        b=-C/B
        P=(0,int(np.round(b)))
        Q=(img.shape[1],int(np.round(m*img.shape[1]+b)))
        img_color=segment_PQ(img, P, Q, color)
    plt.imshow(img_color)
    plt.axis('off')
    plt.title("rectas")
    plt.show()
    return img_color

def circunf(img, centro,radio,color, grosor=1):
    if len(img.shape)==2:#si img es de niveles de gris
        img_color=cv2.merge([img,img,img])
    else:
        img_color=img.copy()
    cv2.circle(img_color, centro, radio, color, grosor)
    plt.imshow(img_color)
    plt.axis('off')
    plt.title("circunf")
    plt.show()
    return img_color

def circunf_ABC(img, circABC,color, grosor=1):
    if len(img.shape)==2:#si img es de niveles de gris
        img_color=cv2.merge([img,img,img])
    else:
        img_color=img.copy()
    [A,B,C]=circABC
    radio= int (np.sqrt(A**2+B**2-4*C)/2)
    centro= (int(-A/2), int(-B/2));
    cv2.circle(img_color, centro, radio, color, grosor)
    # plt.imshow(img_color)
    # plt.axis('off')
    # plt.title("circunf")
    # plt.show()
    return img_color