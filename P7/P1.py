import cv2 as cv
from show_images import show_images as show
import numpy as np


img = cv.imread('./images/cookies.JPG', 0)
bin = ((img < 140)*255).astype('uint8')

# Close to remove noise inside cookies
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (41, 41))
closed = cv.morphologyEx(bin, cv.MORPH_CLOSE, kernel)

# Filter small regions
filtered = np.array(closed)
num_labels, img_labels, stats, cg = cv.connectedComponentsWithStats(closed)
for i, stat in enumerate(stats):
    if(stat[4] < 10000):
        filtered[img_labels == i] = 0

show([img, bin, closed, filtered], ['Original', 'Bin', 'Cerrada', 'Filtrada'])

# Image to print out info
img_out = np.array(img)

# Get contours
contours, hierarchy = cv.findContours(
    filtered, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
for contour in contours:
    # Find fitting ellipse
    (center, axis, angle) = cv.fitEllipse(contour)
    x = int(center[0])
    y = int(center[1])

    # Circles must have a ratio close to 1
    is_circle = 0.8 < (axis[0]/axis[1]) < 1.25

    # Draw circle or square
    if is_circle:
        cv.circle(img_out, (x, y), 50, 255, thickness=-1)
    else:
        cv.rectangle(img_out, (x-50, y-50), (x+50, y+50), 255, thickness=-1)

show([img_out], [f'Resultado ({len(contours)} galletas)'])
