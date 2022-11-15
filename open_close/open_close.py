import cv2 as cv
import numpy as np

cap = cv.VideoCapture("src/wood05.mp4")

while True :

    ret, frame = cap.read()
    resize = cv.resize(frame, (0,0), fx = 0.5, fy = 0.5, interpolation = cv.INTER_AREA)




    cv.imshow("원본", resize)

    key = cv.waitKey(5)
    if key == 27 :
        break
