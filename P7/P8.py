import cv2 as cv

if __name__ == "__main__":
    # img = cv.imread('./images/saw.jpg')
    img = cv.imread('./images/chess.jpg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    output = cv.merge([gray, gray, gray])

    pts = cv.cornerHarris(gray, 2, 7, 0.08)
    pts = cv.dilate(pts, None)
    output[pts > .5] = (0, 255, 0)

    cv.imshow('Output', output)
    cv.waitKey(0)
    cv.destroyAllWindows()
