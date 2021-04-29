import cv2
import numpy as np
from show_images import show_images


# Read the main image
img = cv2.imread('./images/templates/mickey.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# Read the template
template = cv2.imread('./images/templates/template_mickey.png', 0)

# Store width and height of template in w and h
w, h = template.shape[::-1]

# Perform match operations.
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

# Specify a threshold
threshold = 0.9

# Store the coordinates of matched area in a numpy array
loc = np.where(res >= threshold)

img_rgb = np.array(img)
# Draw a rectangle around the matched region.
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h),
                  (0, 0, 255), thickness=10)

# Show the final image with the matched area.
show_images([template, img_rgb], ['Template', 'Result'])
