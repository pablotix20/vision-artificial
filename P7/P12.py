import cv2 as cv
import numpy as np
from show_images import show_images

if __name__ == "__main__":
    input = cv.imread('./images/keys/keys1.jpg', 0)

    bin = ((input < 100)*255).astype('uint8')
    close = cv.morphologyEx(bin, cv.MORPH_CLOSE,
                            cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9)))
    close = cv.morphologyEx(close, cv.MORPH_OPEN,
                            cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9)))

    show_images([input, bin, close], [
                'Input', 'Binary', 'Close'])

    contours, hierarchy = cv.findContours(
        close, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    contours_output = np.zeros(shape=close.shape)
    keys = []
    for i, contour in enumerate(contours):
        # Draw contour to display it
        cv.drawContours(contours_output, contours, i, 255, thickness=5)

        # Get center, axis and angle
        (center, axis, angle) = cv.fitEllipse(contour)

        # Draw object in new image
        key = np.zeros(shape=close.shape)
        cv.drawContours(key, contours, i, 255, thickness=-1)

        # Rotate key
        rotation_matrix = cv.getRotationMatrix2D(
            (key.shape[1], key.shape[0]), angle, 1)
        side = max(key.shape[1], key.shape[0])
        key = cv.warpAffine(key, rotation_matrix,
                            (side*3, side*3)).astype('uint8')

        # Rotate if vertical
        new_cont, _ = cv.findContours(
            key, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        (x, y, w, h) = cv.boundingRect(new_cont[0])
        if h > w:
            key = cv.rotate(key, cv.ROTATE_90_CLOCKWISE)

        # Get bounding rectangle and crop all to same size
        new_cont, _ = cv.findContours(
            key, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        (x, y, w, h) = cv.boundingRect(new_cont[0])
        if i == 0:
            first_w = w
            first_h = h
        key = key[y:y+first_h, x:x+first_w]

        # Rotate if upside down
        new_cont, _ = cv.findContours(
            key, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        (x, y, w, h) = cv.boundingRect(new_cont[0])
        center, _, _ = cv.fitEllipse(new_cont[0])
        if center[0] < x+w/2:
            key = cv.rotate(key, cv.ROTATE_180)

        keys.append(key)

    show_images(keys, range(1, len(keys)+1))

    diffs = []
    diff_vals = []
    diff_strings = []
    first_key = keys[0].astype('int16')
    for i in range(1, len(keys)):
        diff = np.abs(keys[i]-first_key)
        diffs.append(diff)
        diff_val = np.sum(diff)/(255*first_w*first_h)
        diff_vals.append(diff_val)
        res = ('iguales' if diff_val < 0.05 else 'diferentes')
        diff_strings.append(f'Diferencia media (0-1): {diff_val}. Son {res}')

    show_images(diffs, diff_strings)
