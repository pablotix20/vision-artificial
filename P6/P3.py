import cv2 as cv
from show_images import show_images as show
from P2 import sudoku_numbers
import numpy as np

MAX_1_RATIO = 0.6


def classify_num_by_holes(bin, print_imgs=True):
    # Get contours
    contours, hierarchy = cv.findContours(
        bin, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

    # Get number of holes as number of child contours
    hole_num = np.zeros(hierarchy.shape[1])
    for i, contour in enumerate(hierarchy[0]):
        # If contour has parent, sum 1 to parent and set count to -1 to ignore it as a number
        if contour[3] != -1:
            hole_num[contour[3]] += 1
            hole_num[i] = -1

    # Create masks for numbers
    mask_8 = np.zeros(bin.shape, np.uint8)
    mask_469 = np.zeros(bin.shape, np.uint8)
    mask_12357 = np.zeros(bin.shape, np.uint8)
    # Draw numbers on each mask, based on selected contours
    for i, holes in enumerate(hole_num):
        if holes == 0:
            cv.drawContours(mask_12357, contours, i, 255, cv.FILLED)
        elif holes == 1:
            cv.drawContours(mask_469, contours, i, 255, cv.FILLED)
        elif holes == 2:
            cv.drawContours(mask_8, contours, i, 255, cv.FILLED)
        else:
            continue

    # Apply mask to binary image
    mask_8 &= bin
    mask_469 &= bin
    mask_12357 &= bin

    if print_imgs:
        show([bin, mask_8, mask_469, mask_12357], [
            'Input', '8', '4,6,9', '1,2,3,5,7'])

    return (mask_8, mask_469, mask_12357)


def find_ones(bin, print_imgs=True):
    # Get zones and stats
    num_labels, img_labels, stats, cg = cv.connectedComponentsWithStats(bin)

    # Remove all regions below the max ratio (all none 1s)
    for i, region in enumerate(stats):
        ratio = region[2]/region[3]
        if ratio > MAX_1_RATIO:
            img_labels[img_labels == i] = 0
        else:
            img_labels[img_labels == i] = 255

    if print_imgs:
        show([bin, img_labels], ['Input', '1'])

    return img_labels


if __name__ == "__main__":
    img = cv.imread('./images/sudoku.png', 0)
    bin = cv.adaptiveThreshold(
        img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 91, 2)

    # Invert image if necessary, objets must be white
    if np.average(bin) > 127:
        bin = 255-bin

    numbers = sudoku_numbers(bin, False)
    mask_8, mask_469, mask_12357 = classify_num_by_holes(numbers, True)
    mask_1 = find_ones(mask_12357, True)
