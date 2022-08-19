import cv2 as cv
import numpy as np

cap = cv.VideoCapture('C:/Users/sinji/Desktop/firstick/src/A.mp4')

while True :    

    ret, frame = cap.read()
    resize = cv.resize(frame, (0,0), fx = 0.5, fy = 0.5, interpolation = cv.INTER_AREA)

    dst = resize.copy()
    gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(gray, 5000, 1500, apertureSize = 5, L2gradient = False)

    line = cv.HoughLines(canny, 0.8, np.pi / 180 , 150, srn = 100, stn = 200, min_theta = 0, max_theta = np.pi)

    if line is not None :
        # 표준 허프 변환
        for i in line :
            rho, theta = i[0][0], i[0][1]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a*rho, b*rho

            scale = frame.shape[0] + frame.shape[1]

            x1 = int(x0 + scale * -b)
            y1 = int(y0 + scale * a)
            x2 = int(x0 - scale * -b)
            y2 = int(y0 - scale * a)

            cv.line(dst, (x1, y1), (x2, y2), (0,0,255), 2)
            cv.circle(dst, (x0, y0), 3, (255, 0, 0), 5, cv.FILLED)


    cv.imshow("resize", resize)
    cv.imshow("dst", dst)

    key = cv.waitKey(10)
    if key == 27 :
        break