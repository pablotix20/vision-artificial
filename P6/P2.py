import cv2 as cv
from show_images import show_images as show
import numpy as np

MIN_SIZE = 200
MAX_SIZE = 1000
MIN_RATIO = 0.2
MAX_RATIO = 0.8


def sudoku_numbers(img, print_imgs=True):
    # Erode image to avoid fake isolated regions
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    eroded = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    size_filtered = eroded.copy()
    ratio_filtered = eroded.copy()

    # Find regions
    num_labels, img_labels, stats, cg = cv.connectedComponentsWithStats(eroded)

    # Analyze each region
    for i in range(stats.shape[0]):
        # If element or bounding ratio is too big or too small, remove it
        size = stats[i, 4]
        ratio = stats[i, 2]/stats[i, 3]
        if size < MIN_SIZE or size > MAX_SIZE:
            size_filtered[img_labels == i] = 0
        if ratio > MAX_RATIO or ratio < MIN_RATIO:
            ratio_filtered[img_labels == i] = 0

    # Result after applying both filters
    result = size_filtered & ratio_filtered

    if print_imgs:
        show([img, size_filtered, ratio_filtered, result], [
            'Binarizada', 'Filtrado por tamaño', 'Filtrado por ratio', 'Filtrado por tamaño y ratio'])

    return result


if __name__ == "__main__":
    img = cv.imread('./images/sudoku.png', 0)
    bin = cv.adaptiveThreshold(
        img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 91, 2)

    # Invert image if necessary, objets must be white
    if np.average(bin) > 127:
        bin = 255-bin

    sudoku_numbers(bin)
