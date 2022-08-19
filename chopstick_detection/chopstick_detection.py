import cv2 as cv
import numpy as np

cap = cv.VideoCapture('C:/Users/sinji/Desktop/firstick/src/A.mp4')

while True :    

    ret, frame = cap.read()
    resize = cv.resize(frame, (0,0), fx = 0.5, fy = 0.5, interpolation = cv.INTER_AREA)

    dst = resize.copy()
    gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    canny = cv.Canny(blur, 1000, 2000, apertureSize = 5, L2gradient = True)

    line = cv.HoughLinesP(canny, 0.8, np.pi / 180 , 90, minLineLength = 10, maxLineGap = 100)

    if line is not None :
        # 확률적 허프 변환
        for i in line :
            cv.line(dst, (int(i[0][0]), int(i[0][0])), (int(i[0][2]), int(i[0][3])), (0, 0, 255), 2)

    cv.imshow("resize", resize)
    # cv.imshow("gray", gray)
    cv.imshow("canny", canny)
    cv.imshow("dst", dst)

    key = cv.waitKey(10)
    if key == 27 :
        break