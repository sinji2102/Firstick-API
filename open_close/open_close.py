import cv2 as cv
import numpy as np

cap = cv.VideoCapture("src/wood05.mp4")

while True :

    key = cv.waitKey(5)
    if key == 27 :
        break
