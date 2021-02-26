import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.cvtColor(cv2.imread('./images/caja06.png'),
                   cv2.COLOR_BGR2GRAY).astype('uint8')

m_translate = np.float32([[1, 0, 100], [0, 1, 60]])
m_rotate = cv2.getRotationMatrix2D((0, 0), 45, 1.0)
m_translate_inv = np.float32([[1, 0, -100], [0, 1, -60]])

img1 = cv2.warpAffine(img, m_translate, (img.shape[1], img.shape[0]))
img2 = cv2.warpAffine(img1, m_rotate, (img.shape[1], img.shape[0]))
img3 = cv2.warpAffine(img2, m_translate_inv, (img.shape[1], img.shape[0]))

plt.subplot(2, 2, 1), plt.title('Original'), plt.imshow(
    img, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 2), plt.title('Editada 1'), plt.imshow(
    img1, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 3), plt.title('Editada 2'), plt.imshow(
    img2, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 4), plt.title('Editada 3'), plt.imshow(
    img3, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.show()

img4 = cv2.rotate(img3, 0)

plt.subplot(1, 1, 1), plt.title('Rotada'), plt.imshow(
    img4, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.show()
