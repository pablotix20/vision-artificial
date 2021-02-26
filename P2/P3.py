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

# Create a black image, a window and bind the function to window


#img = cv2.imread('../imagenes/caja_persp.png')
img = cv2.imread('./images/casa1.jpg')
img_size = (img.shape[1], img.shape[0])

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image', draw_circle)
while(len(points) < 4):
    cv2.imshow('image', img)
    cv2.waitKey(10)

points_source = np.float32(points)

points_destination_cropped = np.float32(
    [[0, 0], [img_size[0], 0], [img_size[0], img_size[1]], [0, img_size[1]]])
points_destination_full = np.float32(
    [[img_size[0]*.2, img_size[1]*.2], [img_size[0]*.8, img_size[1]*.2], [img_size[0]*.8, img_size[1]*.8], [img_size[0]*.2, img_size[1]*.8]])
cv2.destroyAllWindows()

transform_matrix_crop = cv2.getPerspectiveTransform(
    points_source, points_destination_cropped)
transformed_image_cropped = cv2.warpPerspective(
    img, transform_matrix_crop, img_size)
transform_matrix_full = cv2.getPerspectiveTransform(
    points_source, points_destination_full)
transformed_image_full = cv2.warpPerspective(
    img, transform_matrix_full, img_size)

plt.subplot(2, 2, 1), plt.title('Original'), plt.imshow(
    img), plt.axis(False)
plt.subplot(2, 2, 2), plt.title('Editada 1'), plt.imshow(
    transformed_image_cropped), plt.axis(False)
plt.subplot(2, 2, 3), plt.title('Editada 2'), plt.imshow(
    transformed_image_full), plt.axis(False)
# plt.subplot(2, 2, 4), plt.title('Editada 3'), plt.imshow(
#     img3, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.show()
