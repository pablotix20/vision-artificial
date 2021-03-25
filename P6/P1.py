import cv2 as cv
from numpy.core.defchararray import center
from show_images import show_images as show
import numpy as np

CLOSE_RADIUS = 30

img = cv.imread('./images/calc1.png', 0)
img = cv.rotate(img, cv.ROTATE_180)  # Uncomment to test rotation detection

otsu_threshold, bin = cv.threshold(
    img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Invert image if necessary, objets must be white
if np.average(bin) > 127:
    bin = 255-bin

# Erode image
kernel = cv.getStructuringElement(
    cv.MORPH_ELLIPSE, (CLOSE_RADIUS, CLOSE_RADIUS))
closed = cv.morphologyEx(bin, cv.MORPH_CLOSE, kernel)


# Get component count
count, labels, stats, cg = cv.connectedComponentsWithStats(closed)

# show([img, bin, closed, labels], ['Original', 'Binarizada',
#                                   f'Cerrada radio {CLOSE_RADIUS}', 'Objetos separados'])
if count != 2:
    raise Exception(
        'Less or more than 1 objects found, adjust closing radius?')

# Get contours
contours = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

drawn_contours = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
cv.drawContours(drawn_contours, contours[0], 0, (255, 0, 0), 1)

# Get minimum bounding box for the object, as well as set of points
rect = cv.minAreaRect(contours[0][0])
points_src = cv.boxPoints(rect)

cv.drawContours(drawn_contours, [np.int0(points_src)], 0, (0, 255, 0), 1)

# Get dimensions of the object box
width = int(np.linalg.norm(points_src[1]-points_src[0]))
height = int(np.linalg.norm(points_src[2]-points_src[1]))

# Get equivalent points when transformed
points_dst = np.float32([[0, 0], [width, 0], [width, height]])

# Get transform matrix and do transformation
cropping_matrix = cv.getAffineTransform(points_src[:3], points_dst)
rotated = cv.warpAffine(img, cropping_matrix, (width, height))
rotated_bin = cv.warpAffine(closed, cropping_matrix, (width, height))

if width > height:
    rotated = cv.rotate(rotated, cv.ROTATE_90_CLOCKWISE)
    rotated_bin = cv.rotate(rotated_bin, cv.ROTATE_90_CLOCKWISE)

# Find whether image is upside down, calculate center of mass, when calculator is positioned correctly, center will be below middle point
shrinked = np.sum(rotated_bin, axis=1)
total = 0
weighted = 0
for i in range(shrinked.shape[0]):
    total += shrinked[i]
    weighted += shrinked[i]*i
center = weighted/total/height

if center < 0.5:
    print('Image was upside down, flipping...')
    rotated_fixed = cv.rotate(rotated, cv.ROTATE_180)
else:
    rotated_fixed = rotated

show([img, drawn_contours, rotated, rotated_fixed], ['Original',
                                                     'Deteccion bordes y bounding box', 'Resultado', 'Resultado con orientación corregida'])
