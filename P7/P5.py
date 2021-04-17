import cv2
import numpy as np
import matplotlib.pyplot as plt
from show_images import show_images as show
import math

if __name__ == "__main__":
    img = cv2.imread('./images/bottles/botella17.bmp', 0)
    output = cv2.merge([img, img, img])
    bin = ((img < 100)*255).astype('uint8')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    close = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, kernel)

    sum = np.sum(close, axis=1)
    empy_rows = np.where(sum == 0)[0]
    last_empty_row = empy_rows[-1]
    print(f'Water height is {img.shape[0]-last_empty_row}')

    cv2.line(output, (0, last_empty_row),
             (1000, last_empty_row), color=(255, 0, 0), thickness=2)
    show([img, bin, close, output], ['Original', 'Binary', 'Closed', 'Output'])
