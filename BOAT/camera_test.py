import cv2
import numpy as np

# 選擇第二隻攝影機
# cap = cv2.VideoCapture('http://192.168.21.1:8080/video')
cap = cv2.VideoCapture(0)
# cap2 = cv2.VideoCapture(1)

# 設定擷取的畫面解析度
# cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
# cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
background = cv2.createBackgroundSubtractorMOG2()
while(True):
    # 從攝影機擷取一張影像
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()
    bg = background.apply(frame)
    # 將圖片轉為灰階
    #frame = cv2.cvtColor(frame, background)

    # 顯示圖片
    cv2.imshow('frame', bg)
    # cv2.imshow('frame', frame)
    # cv2.imshow('frame2', frame2)
    # cv2.imshow("Frame", np.hstack([frame, frame2]))

    # 若按下 q 鍵則離開迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝影機
cap.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()
