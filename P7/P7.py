import cv2
import ransac
import lms
import draw
import numpy as np
from show_images import show_images as show

if __name__ == "__main__":
    img = cv2.imread('./images/chavetas/chaveta01.png', 0)
    img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]
    output = np.array(img_bin)

    # Find contours and take outter one
    cnt = cv2.findContours(img_bin, cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_NONE)

    # This function obtains the best set of points defining a circle
    inliers = ransac.ransac_circunf(cnt[0][0], 100)

    # Make a circle from these points
    circABC = lms.circunf(inliers)

    # Draw circle into output image
    output = draw.circunf_ABC(output, circABC, (255, 0, 0), 3)

    show([img, output], ['Input', 'Output'])
