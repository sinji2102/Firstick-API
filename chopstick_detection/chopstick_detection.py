import cv2 as cv
import numpy as np

cap = cv.VideoCapture("C:/Users/sinji/Desktop/firstick/src/iron_wrong01.mp4")
cnt = 1

# 젓가락 전체 길이와 젓가락 쥔 앞 부분 길이 확인해서 1/5 지점 이면 True를 출력하는 코드
while True :    

    ret, frame = cap.read()
    resize = cv.resize(frame, (0,0), fx = 0.5, fy = 0.5, interpolation = cv.INTER_AREA)

    dst = resize.copy()
    gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    canny = cv.Canny(blur, 800, 2000, apertureSize = 5, L2gradient = True)

    # 젓가락 전체
    line = cv.HoughLinesP(canny, 0.2, np.pi / 180 , 50, minLineLength = 400, maxLineGap = 150)

    # 젓가락 일부
    lines = cv.HoughLinesP(canny, 0.2, np.pi / 180, 50, minLineLength = 150, maxLineGap = 55)

    whole = 0
    part = 0

    # 젓가락 전체 출력 / 빨간색
    if line is not None :
        for i in line :
            (a1, b1, a2, b2) = i[0]
            if abs(b1 - b2) < 70 :
                cv.line(resize, (a1, b1), (a2, b2), (0, 0, 255), 2)
                whole = a2 - a1

    # 젓가락 일부(손가락 부분 제거하고) 출력 / 노란색
    if lines is not None :
        for i in lines :
            (a1, b1, a2, b2) = i[0]
            if abs((b2 - b1)/(a2 - a1)) < 5 :  # 각도 왜 안 돼??
                # cv.line(resize, (a1, b1), (a2, b2), (0, 255, 255), 2)
                part = a2 - a1
    
    if whole * 0.5 < part < whole * 0.6 :
        print("True", cnt)
        cnt += 1

    cv.imshow("resize", resize)
    cv.imshow("canny", canny)
    # cv.imshow("dst", dst)

    key = cv.waitKey(5)
    if key == 27 :
        break
