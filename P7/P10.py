import cv2 as cv
import numpy as np
from show_images import show_images


def indices_array_generic(m, n):
    r0 = np.arange(m)  # Or r0,r1 = np.ogrid[:m,:n], out[:,:,0] = r0
    r1 = np.arange(n)
    out = np.empty((m, n, 2), dtype=int)
    out[:, :, 0] = r0[:, None]
    out[:, :, 1] = r1
    return out


if __name__ == "__main__":
    input = cv.resize(cv.imread('./images/plates/plate_1.jpg'), (960, 720))
    # input = cv.resize(cv.imread('./images/plates/plate_2.jpg'), (960, 720))
    # input = cv.resize(cv.imread('./images/plates/plate_3.jpg'), (960, 720))

    input = cv.cvtColor(input, cv.COLOR_BGR2RGB)

    blue = input[:, :, 2]

    binary = cv.adaptiveThreshold(blue, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv.THRESH_BINARY, 201, 2)
    closing = cv.morphologyEx(
        binary, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
    opening = cv.morphologyEx(
        closing, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9)))

    size_filtered = np.array(opening)
    num_labels, img_labels, stats, cg = cv.connectedComponentsWithStats(
        opening)
    for i, stat in enumerate(stats):
        if i == 0:
            continue
        # Size filter
        if stat[4] < 20000:
            size_filtered[img_labels == i] = 0
        # Aspect ratio filter
        elif not(1 < abs(stat[2]/stat[3]) < 3):
            size_filtered[img_labels == i] = 0

    # Filter by contour specs
    contour_filtered = np.zeros(shape=size_filtered.shape)
    num_labels, img_labels, stats, cg = cv.connectedComponentsWithStats(
        size_filtered)
    for i, stat in enumerate(stats):
        if i == 0:
            continue
        (x, y, w, h) = stat[:4]
        part = np.zeros(shape=size_filtered.shape, dtype='uint8')
        part[img_labels == i] = 255

        # Get contours
        contours, hierarchy = cv.findContours(
            part, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        # Check perimeter/area ratio
        perimeter = cv.arcLength(contours[0], True)
        area = cv.contourArea(contours[0])
        ratio = perimeter/area
        if 0.01 < ratio < 0.025:
            print('Plate found')
            contour_filtered[img_labels == i] = 255
            ellipse = cv.fitEllipse(contours[0])
            break

    # Calculate distances to center
    x = int(ellipse[0][0])
    y = int(ellipse[0][1])
    h = int(ellipse[1][0])
    w = int(ellipse[1][1])
    center = np.array([y, x], dtype='int')
    distances = indices_array_generic(
        contour_filtered.shape[0], contour_filtered.shape[1])
    distances = np.sum((distances-center)**2, axis=2)**.5
    distances_filtered = distances*contour_filtered/255

    # Find largest distance inside each quadrant
    bounds = np.zeros((4, 2), dtype='int32')

    quadrant = distances_filtered[:y, :x]
    bounds[0] = np.unravel_index(
        np.argmax(quadrant, axis=None), quadrant.shape)

    quadrant = distances_filtered[:y, x:]
    bounds[1] = np.unravel_index(
        np.argmax(quadrant, axis=None), quadrant.shape)
    bounds[1] += [0, x]

    quadrant = distances_filtered[y:, x:]
    bounds[2] = np.unravel_index(
        np.argmax(quadrant, axis=None), quadrant.shape)
    bounds[2] += [y, x]

    quadrant = distances_filtered[y:, :x]
    bounds[3] = np.unravel_index(
        np.argmax(quadrant, axis=None), quadrant.shape)
    bounds[3] += [y, 0]

    bounds = np.flip(bounds, axis=1)

    for i in range(4):
        cv.circle(distances_filtered, tuple(bounds[i]),
                  radius=15, color=(127), thickness=-1)

    # Obtain transform matrix
    points_source = np.float32(bounds)
    points_dest = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype='float32')
    transform_matrix = cv.getPerspectiveTransform(points_source, points_dest)

    transformed = cv.warpPerspective(input, transform_matrix, (w, h))

    show_images([input, blue, binary, opening, size_filtered, contour_filtered, distances_filtered, transformed], [
                'Input', 'Blue', 'Binary', 'Closed & opened', 'Size & aspect filtered', 'Contour filtered', 'Distances & points', 'Transformed'])
