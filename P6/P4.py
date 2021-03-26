import cv2 as cv
from show_images import show_images as show
from P2 import sudoku_numbers
import numpy as np
import pandas as pd
import math

# Obtained after first execution
NUM_MAP = [-1, 6, 4, 7, 7, 9, 6, 8, 5, 7, 2, 3,
           9, 8, 5, 4, 3, 1, 7, 5, 2, 3, 2, 8, 2, 3, 1]


def get_contour_specs(bin):
    # Get contours
    contours, hierarchy = cv.findContours(
        bin, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

    # Get number of holes as number of child contours
    hole_num = np.zeros(hierarchy.shape[1], 'int8')
    for i, contour in enumerate(hierarchy[0]):
        # If contour has parent, sum 1 to parent and set count to -1 to ignore it as a number
        if contour[3] != -1:
            hole_num[contour[3]] += 1
            hole_num[i] = -1

    # Ensure there is only 1 parent contour
    for i, count in enumerate(hole_num):
        if i != 0 and count != -1:
            raise Exception('More than 1 contour was found')

    perimeter = cv.arcLength(contours[0], True)

    return hole_num[0], perimeter


if __name__ == "__main__":
    img = cv.imread('./images/sudoku.png', 0)
    bin = cv.adaptiveThreshold(
        img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 91, 2)

    # Invert image if necessary, objets must be white
    if np.average(bin) > 127:
        bin = 255-bin

    # Get processed numbers
    numbers = sudoku_numbers(bin, False)

    # Copy to show region numbers
    labeled_bin = cv.cvtColor(numbers, cv.COLOR_GRAY2RGB)

    # Find regions
    num_labels, img_labels, stats, cg = cv.connectedComponentsWithStats(
        numbers)

    number_stats = pd.DataFrame(columns=[
                                'region', 'number', 'width', 'height', 'bounding_area', 'perimeter', 'area', 'ratio', 'center_x', 'center_y', 'occupation', 'circularity', 'hole_count'])
    number_stats.set_index('region', inplace=True)

    # Analyze each region
    for i, stat in enumerate(stats):
        if i == 0:
            continue

        masked_num = numbers & (img_labels == i)
        x, y, w, h, area = stat
        ratio = w/h
        hole_count, perimeter = get_contour_specs(masked_num)
        circularity = 4*math.pi*area/(perimeter**2)
        bounding_area = w*h
        moments = cv.moments(masked_num)
        cx = (moments['m10']/moments['m00']-x)/w
        cy = (moments['m01']/moments['m00']-y)/h
        occupation = area/bounding_area

        number_stats.loc[i] = [NUM_MAP[i], w, h, bounding_area, perimeter, area, ratio,
                               cx, cy, occupation, circularity, hole_count]

        cv.rectangle(labeled_bin, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv.putText(labeled_bin, f'{i}', (x, y),
                   cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

    print(number_stats)
    number_stats.to_excel('./output/number_stats.xlsx')

    show([labeled_bin], ['Labeled numbers'])
