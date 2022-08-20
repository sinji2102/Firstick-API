import cv2 as cv
import numpy as np

cap = cv.VideoCapture('C:/Users/sinji/Desktop/firstick/src/A.mp4')

while True :    

    ret, frame = cap.read()
    resize = cv.resize(frame, (0,0), fx = 0.5, fy = 0.5, interpolation = cv.INTER_AREA)

    dst = resize.copy()
    gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    canny = cv.Canny(blur, 700, 2000, apertureSize = 5, L2gradient = True)

    line = cv.HoughLinesP(canny, 0.5, np.pi / 180 , 30, minLineLength = 5, maxLineGap = 30)

    arr = []
    x11, y11, x12, y12, x21, y21, x22, y22 = 0, 0, 0, 0, 0, 0, 0, 0

    if line is not None :
        for i in line :
            (a1, b1, a2, b2) = i[0]
            if a1 > 310 :   # 빨간색 라인
                cv.line(resize, (a1, b1), (a2, b2), (0, 0, 255), 2)
                x11 = a1
                y11 = b1
                x12 = a2
                y12 = b2


    cv.imshow("resize", resize)
    # cv.imshow("gray", gray)
    cv.imshow("canny", canny)
    cv.imshow("dst", dst)

    key = cv.waitKey(10)
    if key == 27 :
        break
