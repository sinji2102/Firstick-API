import cv2 as cv
import numpy as np

cap = cv.VideoCapture('C:/Users/sinji/Desktop/firstick/src/B.mp4')

while True :    

    ret, frame = cap.read()
    resize = cv.resize(frame, (0,0), fx = 0.5, fy = 0.5, interpolation = cv.INTER_AREA)

    dst = resize.copy()
    gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    canny = cv.Canny(blur, 800, 2000, apertureSize = 5, L2gradient = True)

    line = cv.HoughLinesP(canny, 0.2, np.pi / 180 , 30, minLineLength = 400, maxLineGap = 150)

    if line is not None :
        for i in line :
            (a1, b1, a2, b2) = i[0]
            if abs(b1 - b2) < 70 :
                cv.line(resize, (a1, b1), (a2, b2), (0, 0, 255), 2)

    cv.imshow("resize", resize)
    cv.imshow("canny", canny)
    # cv.imshow("dst", dst)

    key = cv.waitKey(5)
    if key == 27 :
        break
