import cv2 as cv
from show_images import show_images as show
import numpy as np
import math
import os


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename), 0)
        if img is not None:
            images.append(img)
    return images


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


def count_objects(img):
    # Find regions
    num_labels, img_labels, stats, cg = cv.connectedComponentsWithStats(bin)

    count = {'tornillo': 0, 'tuerca': 0, 'arandela': 0, 'llave_abierta': 0,
             'llave_cerrada': 0, 'brida': 0, 'pinon': 0, 'corona': 0}

    # Analyze each region
    for i, stat in enumerate(stats):
        if i == 0:
            continue

        # Calculate parameters
        masked_num = bin & (img_labels == i)
        x, y, w, h, area = stat

        # Ignore fake objects
        if area < 100:
            continue

        hole_count, perimeter = get_contour_specs(masked_num)
        circularity = 4*math.pi*area/(perimeter**2)
        bounding_area = w*h
        occupation = area/bounding_area

        # Decide which object type it is
        if hole_count == 0:
            if circularity > 0.2:
                count['tornillo'] += 1
            else:
                count['llave_abierta'] += 1
        elif hole_count == 1:
            if occupation < 0.4:
                count['arandela'] += 1
            else:
                if circularity > 0.5:
                    count['tuerca'] += 1
                else:
                    count['pinon'] += 1
        elif hole_count == 2:
            count['llave_cerrada'] += 1
        elif hole_count == 7:
            count['brida'] += 1
        elif hole_count > 7:
            count['corona'] += 1

    return count


if __name__ == "__main__":
    imgs = load_images_from_folder('./images/tools/')

    images = []
    labels = []
    for img in imgs:
        bin = ((img < 100)*255).astype('uint8')
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        bin = cv.morphologyEx(bin, cv.MORPH_CLOSE, kernel)

        images.append(bin)
        count = count_objects(bin)
        label = ''
        for a in count:
            if count[a] > 0:
                label += f'{count[a]} {a}, '
        labels.append(label)

    show(images, labels)
