import cv2 as cv
from show_images import show_images as show
import numpy as np

ER_RADIUS = 20


def countObjects(img, radius, connectivity):
    # Select kernel type
    if connectivity == 4:
        shape = cv.MORPH_ELLIPSE
    elif connectivity == 8:
        shape = cv.MORPH_RECT
    else:
        raise Exception('Invalid connectivity')

    # Erode image
    kernel = cv.getStructuringElement(shape, (radius, radius))
    eroded = cv.erode(img, kernel)

    # Get component count
    ret, labels = cv.connectedComponents(eroded)

    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0

    return (ret-1, eroded, labeled_img)


img = cv.imread('./images/screws.png', 0)
otsu_threshold, bin = cv.threshold(
    img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Inverse image if necessary, objets must be white
if np.average(bin) > 127:
    bin = 255-bin

count, eroded, labels = countObjects(bin, ER_RADIUS, 4)

show([img, bin, eroded, labels], ['Original', 'Binarizada',
                                  f'Erosionada radio {ER_RADIUS}', f'Objetos separados ({count})'])
