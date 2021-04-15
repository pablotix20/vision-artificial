import cv2
import numpy as np
import matplotlib.pyplot as plt
from show_images import show_images as show
import math

if __name__ == "__main__":
    img = cv2.imread('./images/lanes.jpg', 0)
    output = img_color = cv2.merge([img, img, img])
    output_filter = np.array(output)
    edges = cv2.Canny(img, 300, 600)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    print(img.shape)

    print(f'{len(lines)} lines detected')
    for i, line in enumerate(lines):
        rho, theta = line[0]

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(output, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Discard line if not desired
        # Detect vertical lines
        if theta == 0:
            # Remove only if is on the borders of the image
            if rho < 5 or abs(rho-img.shape[1]) < 5:
                continue
        # Discard horizontal lines
        elif abs(theta - math.pi/2) < 0.1:
            continue
        # If image is close to another one not yet printed, discard
        skip = False
        for (n_rho, n_theta) in lines[i+1:, 0]:
            if abs(rho-n_rho) < 100 and abs(theta-n_theta) < 0.1:
                skip = True
        if skip:
            continue

        cv2.line(output_filter, (x1, y1), (x2, y2), (255, 0, 0), 2)

    show([img, edges, output, output_filter], ['Original',
                                               'Borders', 'Hough lines', 'Hough lines filtered'])
