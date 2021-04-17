# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 19:59:36 2021
Funciones para calcular recta y circunferencia por minimos cuadrados LMS
@author: eusebio
"""

import cv2
import numpy as np


def recta( puntos ):
    '''Ajusta por minimos cuadrados a recta Ax + By + C = 0 calcula eje inercia
    Devuelve [A,B,C]. n(A,B) unitario. Viene bien para calculo dist(P,r)
    x,y contiene las x (columnas)y las y (filas) de los pix de cont'''

    #(vx, vy, x0, y0), where (vx, vy) is a normalized vector collinear to 
    #the line and (x0, y0) is a point on the line.
    [vx, vy, x, y] = cv2.fitLine(puntos, cv2.DIST_L2, param=0, reps=0.01, aeps=0.01)
    
    m= vy[0]/vx[0]
    b=y[0]-m*x[0];
    #mx-y+b=0. 
    A=m; B=-1; C=b;
    
    #Normalizo n(A,B)=1
    n=[A,B];
    normAB=np.linalg.norm(n);
    A = A/normAB;
    B = B/normAB;
    C = C/normAB;

    return [A,B,C]

def circunf(puntos):
    '''minimos cuadrados circunferencia calculando la pseudoinversa
    Devuelve los parametros A,B,C de la circunferencia x2+y2+Ax+By+C=0
    que ajusta los puntos'''

    n=len(puntos)
    A=np.ones( (n,3), np.float32)
    
    for i in range(n):
        A[i][0]= puntos[i][0][0]
        A[i][1]= puntos[i][0][1]

    b=np.ones( (n,1), np.float32)
    for i in range(n):
        b[i][0] = - puntos[i][0][0]**2 - puntos[i][0][1]**2 


    # The pseudo-inverse of a matrix A, denoted A^+, is defined as: 
    # the matrix that ‘solves’ [the least-squares problem] Ax = b,
    # i.e., if \bar{x} is said solution, then A^+ is that matrix such that x = A^+ b.        
    
    pseudo_inv_A= np.linalg.pinv(A)

    [A,B,C]=pseudo_inv_A@b;

    return [A,B,C]