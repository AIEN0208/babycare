#coding=utf-8
import cv2
import os
import numpy as np

cap = cv2.VideoCapture(1)
if cap.read()[1] is None:
    cap = cv2.VideoCapture(0)
i = 0
over = 0
width = int(cap.get(3))  # float
height = int(cap.get(4))  # float
resultFrame = np.empty((height, width), dtype=np.uint8)
resultFrame.fill(255)
resultFrame = cv2.cvtColor(resultFrame, cv2.COLOR_GRAY2BGR)


while 1:
    _,image = cap.read();
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        while 1:
            #构造一个3×3的结构元素 
            element = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
            dilate = cv2.dilate(image, element)
            erode = cv2.erode(image, element)
            
            #将两幅图像相减获得边，第一个参数是膨胀后的图像，第二个参数是腐蚀后的图像
            result = cv2.absdiff(dilate,erode);
            
            #上面得到的结果是灰度图，将其二值化以便更清楚的观察结果
            retval, result = cv2.threshold(result, 40, 255, cv2.THRESH_BINARY); 
            #反色，即对二值图每个像素取反
            result = cv2.bitwise_not(result); 
            if i < 10:
                resultFrame = np.minimum(resultFrame, result)
                i+=1

            resf = cv2.flip(resultFrame, 1)
            cv2.imshow('final', resf)
            # cv2.imshow('final', np.hstack([resf,redmasked,yellowmasked,greenmasked]))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                over = 1
                break
    image = np.uint8(np.clip((1.5 * image + 10), 0, 255))
    image = cv2.flip(image, 1)
    cv2.imshow('Normal', image)
    key = cv2.waitKey(1) & 0xFF
    if over:
        break
# resf = cv2.cvtColor(resf, cv2.COLOR_GRAY2BGR)
# cv2.imshow('RGB',resf)
cv2.imwrite('sketch.jpg', resf)
cap.release()
cv2.destroyAllWindows()

def next():
    os.system("python3 painting2.py")
next()
