import cv2
import ransac
import lms
import draw
import numpy as np
from show_images import show_images as show
from noiselib import add_salt_and_pepper_noise

if __name__ == "__main__":
    img = cv2.imread('./images/chavetas/chaveta01.png', 0)
    noisy = add_salt_and_pepper_noise(img, density=0)
    # noisy = add_salt_and_pepper_noise(img, density=0.01)
    img_bin = cv2.threshold(noisy, 0, 255, cv2.THRESH_OTSU)[1]
    output = cv2.merge([noisy, noisy, noisy])

    # Find contours and take outter one
    cnt = cv2.findContours(img_bin, cv2.RETR_TREE,
                           cv2.CHAIN_APPROX_NONE)

    # Concatenate all contours to find circles everywhere
    contours = np.concatenate(cnt[0])

    # This function obtains the best set of points defining a circle
    inliers = ransac.ransac_circunf(contours, 100, max_size=140)

    # Make a circle from these points
    circABC = lms.circunf(inliers)

    # Draw circle into output image
    output = draw.circunf_ABC(output, circABC, (255, 0, 0), 3)

    show([img, output], ['Input', 'Output'])
