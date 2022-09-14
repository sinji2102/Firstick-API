import cv2 as cv
import numpy as np

cap = cv.VideoCapture("C:/Users/sinji/Desktop/firstick/src/wood05.mp4")
cnt = 1

# 젓가락 전체 길이와 젓가락 쥔 앞 부분 길이 확인해서 1/5 지점 이면 True를 출력하는 코드 >> 값 수정 + 1/5 지점에 점 찍는 것까지
while True :    

    ret, frame = cap.read()
    resize = cv.resize(frame, (0,0), fx = 0.5, fy = 0.5, interpolation = cv.INTER_AREA)

    dst = resize.copy()
    gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    blur_ = cv.GaussianBlur(gray, (3, 3), 0)
    canny = cv.Canny(blur, 800, 2000, apertureSize = 5, L2gradient = True)
    canny_ = cv.Canny(blur_, 300, 1700, apertureSize = 5, L2gradient = True)

    # 젓가락 전체
    line = cv.HoughLinesP(canny, 0.2, np.pi / 180 , 50, minLineLength = 400, maxLineGap = 150)

    # 젓가락 일부
    lines = cv.HoughLinesP(canny_, 0.2, np.pi / 180, 50, minLineLength = 150, maxLineGap = 40)

    whole = 0
    part = 0

    a11, a12, a21, a22, b11, b12, b21, b22 = 0, 0, 0, 0, 0, 0, 0, 0

    # 젓가락 전체 출력 / 빨간색
    if line is not None :
        for i in line :
            (a11, b11, a12, b12) = i[0]
            if abs(b11 - b12) < 70 :
                cv.line(resize, (a11, b11), (a12, b12), (0, 0, 255), 2)
                whole = a12 - a11

    # 젓가락 일부(손가락 부분 제거하고) 출력 / 노란색
    if lines is not None :
        for i in lines :
            (a21, b21, a22, b22) = i[0]
            if abs((b22 - b21)/(a22 - a21)) < 5 :
                # cv.line(resize, (a21, b21), (a22, b22), (0, 255, 255), 2)
                part = a22 - a21

    # 점 찍기
    if lines is not None :
        if whole * 0.55 < part < whole * 0.7 :
            print("True", cnt)
            cnt += 1
            cv.line(resize, (a22, b22), (a22, b22), (255, 0, 0), 15)

    cv.imshow("resize", resize)
    cv.imshow("canny", canny)
    # cv.imshow("dst", dst)

    key = cv.waitKey(5)
    if key == 27 :
        break
