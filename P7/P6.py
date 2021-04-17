import cv2
from show_images import show_images as show

if __name__ == "__main__":
    img = cv2.imread('./images/balls.jpg', 0)
    # img = cv2.imread('./images/chavetas/chaveta01.png', 0)
    output = cv2.merge([img, img, img])

    lines = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.5, 20)

    if lines is not None:
        print(f'{len(lines)} circles detected')
        for i, line in enumerate(lines[0]):
            (x, y, r) = line

            cv2.circle(output, (x, y), int(r), (255, 0, 0), thickness=5)
    else:
        print('No circles detected')

    show([img,  output], ['Original', 'Output'])
