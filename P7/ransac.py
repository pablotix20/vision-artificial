# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 20:06:31 2021

@author: eusebio
"""

import numpy as np
import lms
import points as pts


def inliers_recta(ptos_cnt, A, B, C, tolerancia):
    ''' Devuelve inliers para recta Ax2+Bx+C=0
    Supone n(A,B) unitario para calculo dist(P,r) = abs(A*x + B*y + C)
    normAB=sqrt(A^2 + B^2)=1'''

    inliers = []
    num_ptos = len(ptos_cnt)
    num_inliers = 0
    for k in range(num_ptos):
        x = ptos_cnt[k][0][0]
        y = ptos_cnt[k][0][1]
        dist_punto_recta = abs(A*x + B*y + C)
        if dist_punto_recta < tolerancia:  # Si el pixel se ajusta a la recta
            num_inliers = num_inliers + 1
            inliers.append(ptos_cnt[k])

    return np.array(inliers)


def inliers_circunf(ptos_cnt, A, B, C, tolerancia):
    ''' Devuelve inliers para circunferencia x2+y2+ Ax+By+C=0'''
    radio = np.sqrt(A**2+B**2-4*C)/2
    centro = ((-A/2), int(-B/2))
    # franja de consenso (R1,R2)
    R1 = radio-tolerancia
    R2 = radio+tolerancia
    inliers = []
    num_ptos = len(ptos_cnt)
    num_inliers = 0
    for k in range(num_ptos):
        x = ptos_cnt[k][0][0]
        y = ptos_cnt[k][0][1]
        dist_centro = np.sqrt((x-centro[0])**2 + (y-centro[1])**2)

        if R1 < dist_centro < R2:  # Si el pixel esta en la corona
            num_inliers = num_inliers + 1
            inliers.append(ptos_cnt[k])

    return np.array(inliers)


def ransac_recta(puntos, num_max_iter):
    '''puntos(x,y) contiene las x (columnas)y las y (filas) de los pix de cont
    num_max_iter numero máximo de iteraciones
    mejor_inliers son los puntos del mejor modelo encontrado
    inliers hay que pasarlos LMS para afinar recta
    devuelve mejor_inliers
    '''

    tolerancia = 1  # distancia max a recta para que sea inlier
    num_inliers = 0
    mejor_num_inliers = 0  # el numero de inliers más alto q se ha encontrado

    for i in range(num_max_iter):
        samples = pts.get_random_points(puntos, 2)

        # recta que pasa por estos dos puntos
        [A, B, C] = lms.recta(samples)

        inliers = inliers_recta(puntos, A, B, C, tolerancia)

        num_inliers = len(inliers)
        if num_inliers > mejor_num_inliers:
            mejor_num_inliers = num_inliers
            mejor_inliers = inliers

    return mejor_inliers


def ransac_circunf(puntos, num_max_iter, max_size=-1, min_size=-1):
    '''puntos(x,y) contiene las x (columnas)y las y (filas) de los pix de cont
    num_max_iter numero máximo de iteraciones
    mejor_inliers son los puntos del mejor modelo encontrado
    inliers hay que pasarlos LMS para afinar circunferencia
    devuelve mejor_inliers
    '''

    tolerancia = 1  # anchura de la corona circular donde estan los inliers
    num_inliers = 0
    mejor_num_inliers = 0  # el numero de inliers más alto q se ha encontrado

    for i in range(num_max_iter):
        samples = pts.get_random_points(puntos, 3)  # cogemos 3 puntos

        # circunferencia que pasa por estos tres puntos
        [A, B, C] = lms.circunf(samples)

        inliers = inliers_circunf(puntos, A, B, C, tolerancia)

        radio = np.sqrt(A**2+B**2-4*C)/2
        # comprobar si el radio está dentro de los parámetros admitidos
        if max_size != -1 and radio > max_size:
            inliers = samples
        if min_size != -1 and radio < min_size:
            inliers = samples

        num_inliers = len(inliers)
        if num_inliers > mejor_num_inliers:
            mejor_num_inliers = num_inliers
            mejor_inliers = inliers

    return mejor_inliers
