import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# models.py에서 생성한 모델 불러오기
with open('test.pkl', 'rb') as f:
    model = pickle.load(f)


# webCam으로 테스트
cap = cv2.VideoCapture(0)

# video로 테스트
# cap = cv2.VideoCapture('./data/open_iron.mp4')
# cap = cv2.VideoCapture('./data/iron_1.mp4')


with mp_hands.Hands(
  max_num_hands=1, # 손은 하나만 인식
  min_detection_confidence=0.5,
  min_tracking_confidence=0.5) as hands:

  while cap.isOpened():
    success, image = cap.read()

    if not success:
      print("Ignoring empty camera frame.")
      # break
      # Webcam으로 테스트시 continue
      continue
    
    # image = cv2.resize(image, (480, 720)) # 영상 데이터일때 resize (3:4 비율 고정)

    # WebCam에서 flip이 필요할 경우 주석 해제
    # image = cv2.flip(image, 0)
    # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
      hand = results.multi_hand_landmarks
      a = [landmarks.landmark for landmarks in hand]
      b = [data for data in a]

      hand_row = []
      x = []
      y = []
      z = []
      for landmarks in hand:
          for data in landmarks.landmark:
            hand_row.append(data.x)
            hand_row.append(data.y)
            hand_row.append(data.z)
            x.append(np.round(data.x, 5))
            y.append(np.round(data.y, 5))
            z.append(np.round(data.z, 5))
          mp_drawing.draw_landmarks(
              image, landmarks, mp_hands.HAND_CONNECTIONS)

      hand_row += [y[16]-y[12],y[15]-y[11], y[16]-y[8], y[15]-y[7]]
      def tmp(ip):
        return np.round(ip, 4)

      # debugging
      print(tmp(y[16]-y[12]),tmp(y[15]-y[11]), tmp(y[16]-y[8]), tmp(y[15]-y[7]))
      print(tmp(x[16]-x[12]))

      # predict
      X = pd.DataFrame([hand_row])
      hand_class = model.predict(X)[0] # prednp
      hand_prob = np.max(model.predict_proba(X)[0])


      # 결과 출력
      print(hand_class, hand_prob) # debugging
      if hand_prob > 0.75:
        if hand_prob <= 0.89: 
          # print('Hmm')
          if hand_class == 'open':
            cv2.putText(image, 'close', (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255),thickness=3)
          else: 
            cv2.putText(image, 'open', (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255),thickness=3)

        else: 
          # print(hand_class)
          cv2.putText(image, hand_class, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255),thickness=3)
      else: 
          cv2.putText(image, 'Hold the chopsticks.', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255),thickness=3)
      cv2.putText(image, hand_class+' / '+str(hand_prob)+'%', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),thickness=3)

    except:
      # print('except')
      pass

    cv2.imshow('predict_test', image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
      break

cap.release()