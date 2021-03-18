import cv2 as cv
from show_images import show_images as show
import numpy as np


def sudoku_numbers(img):
    # Erode image to get horizontal/vertical lines
    kernel_h = cv.getStructuringElement(cv.MORPH_RECT, (45, 1))
    kernel_v = cv.getStructuringElement(cv.MORPH_RECT, (1, 45))
    eroded_h = cv.morphologyEx(img, cv.MORPH_OPEN, kernel_h)
    eroded_v = cv.morphologyEx(img, cv.MORPH_OPEN, kernel_v)
    eroded_matrix = eroded_v | eroded_h

    # Remove matrix from image
    numbers = img-eroded_matrix

    # Open image to remove noise and residual lines
    numbers_eroded = cv.morphologyEx(
        numbers, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4)))

    show([img, eroded_matrix, numbers, numbers_eroded], [
         'Binarizada', 'Matriz erosionada', 'Numeros', 'Numeros filtrados'])

    return numbers_eroded


img = cv.imread('./images/sudoku.png', 0)
bin = cv.adaptiveThreshold(
    img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 91, 2)

# Inverse image if necessary, objets must be white
if np.average(bin) > 127:
    bin = 255-bin

sudoku_numbers(bin)
