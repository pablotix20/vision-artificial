import cv2 as cv
from show_images import show_images as show
import numpy as np
import math
from matplotlib import pyplot as plt


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


def get_object_specs(img):
    # Find regions
    num_labels, img_labels, stats, cg = cv.connectedComponentsWithStats(bin)

    # Array to hold all results
    results = np.zeros(shape=(len(stats)-1, 3))
    centers = []

    # Analyze each region
    for i, stat in enumerate(stats):
        if i == 0:
            continue

        # Calculate parameters
        masked_num = bin & (img_labels == i)
        x, y, w, h, area = stat

        hole_count, perimeter = get_contour_specs(masked_num)
        circularity = 4*math.pi*area/(perimeter**2)
        bounding_area = w*h
        occupation = area/bounding_area
        results[i-1] = [hole_count, circularity, occupation]

        centers.append((int(x+w/2), int(y+h/2)))

    return results, centers


COLOR_MAP = ['c', 'm', 'r', 'g', 'b']
COLOR_MAP_RGB = [(0, 255, 255), (255, 0, 255),
                 (255, 0, 0), (0, 255, 0), (0, 0, 255)]

if __name__ == "__main__":
    img = cv.imread('./images/tools/tuercas_tornillos3.bmp', 0)
    bin = ((img < 100)*255).astype('uint8')

    class_count = int(input('Enter number of classes (1-5): '))

    # Get specs, which will be points for Kmeans, including hole count, circularity and occupation
    specs, centers = get_object_specs(bin)
    specs = specs.astype('float32')
    print(specs)

    # Apply Kmeans
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    ret, label, center = cv.kmeans(
        specs, class_count, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # Plot result points in 3D, whith each label in different colour
    label = label.ravel()
    figure = plt.figure()
    sp = plt.subplot(1, 2, 1, projection='3d')
    for i in range(class_count):
        mask = label == i
        sp.scatter(specs[mask, 0], specs[mask, 1],
                   specs[mask, 2], c=COLOR_MAP[i])
    sp.scatter(center[:, 0], center[:, 1], center[:, 2], c='y')

    # Print image with labelling
    labeled = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    for i, center in enumerate(centers):
        cv.circle(labeled, center, 4, COLOR_MAP_RGB[label[i]], thickness=-1)
    plt.subplot(1, 2, 2).imshow(labeled)

    plt.show()
