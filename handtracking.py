import mediapipe as mp 
import cv2 as c
import time as tm

cap = c.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(False)
mpDraw = mp.solutions.drawing_utils

while True:
    sucess,img = cap.read()
    imgRGB = c.cvtColor(img,c.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    # print(result.multi_hand_landmarks)

    if result.multi_hand_landmarks:
        for handsLms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handsLms,mpHands.HAND_CONNECTIONS)

    c.imshow("Image", img)
    c.waitKey(1)
