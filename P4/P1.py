import cv2
from matplotlib import pyplot as plt
airbus = cv2.imread('./images/cojinetes.bmp', cv2.IMREAD_GRAYSCALE)
airbus_th, airbus_bin = cv2.threshold(airbus, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
perritos = cv2.imread('./images/perritos.jpg', cv2.IMREAD_GRAYSCALE)
perritos_th, perritos_bin = cv2.threshold(perritos, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


plt.subplot(2, 2, 1), plt.title('Original'), plt.imshow(
    airbus, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 2), plt.title(f'Binarizacion auto, threshold {airbus_th}'), plt.imshow(
    airbus_bin, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 3), plt.title('Original'), plt.imshow(
    perritos, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.subplot(2, 2, 4), plt.title(f'Binarizacion auto, threshold {perritos_th}'), plt.imshow(
    perritos_bin, 'gray', vmin=0, vmax=255), plt.axis(False)
plt.show()