import cv2
import numpy as np
import matplotlib.pyplot as plt

points = []

# mouse callback function


def draw_circle(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 5, (255, 0, 255), 0)
        points.append([x, y])


img = cv2.imread('./images/induva.jpg')
img_size = (img.shape[1], img.shape[0])

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image', draw_circle)
while len(points) < 4:
    cv2.imshow('image', img)
    cv2.waitKey(10)
cv2.destroyAllWindows()

points_source = np.float32(points)
points_destination_cropped = np.float32(
    [[0, 0], [img_size[0], 0], [img_size[0], img_size[1]], [0, img_size[1]]])
points_destination_full = np.float32(
    [[img_size[0]*.25, img_size[1]*.25], [img_size[0]*.75, img_size[1]*.25], [img_size[0]*.75, img_size[1]*.75], [img_size[0]*.25, img_size[1]*.75]])

transform_matrix_crop = cv2.getPerspectiveTransform(
    points_source, points_destination_cropped)
transformed_image_cropped = cv2.cvtColor(cv2.warpPerspective(
    img, transform_matrix_crop, img_size), cv2.COLOR_BGR2RGB)
transform_matrix_full = cv2.getPerspectiveTransform(
    points_source, points_destination_full)
transformed_image_full = cv2.cvtColor(cv2.warpPerspective(
    img, transform_matrix_full, img_size), cv2.COLOR_BGR2RGB)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.subplot(2, 2, 1), plt.title('Original'), plt.imshow(
    img_rgb), plt.axis(False)
plt.subplot(2, 2, 2), plt.title('Recorte'), plt.imshow(
    transformed_image_cropped), plt.axis(False)
plt.subplot(2, 2, 3), plt.title('Vista global'), plt.imshow(
    transformed_image_full), plt.axis(False)
plt.show()

plt.subplot(1, 1, 1), plt.title('Recorte'), plt.imshow(
    transformed_image_cropped), plt.axis(False)
plt.show()
plt.subplot(1, 1, 1), plt.title('Vista global'), plt.imshow(
    transformed_image_full), plt.axis(False)
plt.show()
