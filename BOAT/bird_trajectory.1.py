# import the necessary packages
import argparse
import cv2
import numpy as np
from imutils.video import VideoStream


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video", required=True, help="path to input video file")
ap.add_argument("-o", "--output", required=True, help="path to output png")
args = vars(ap.parse_args())
try:
    # cap = VideoStream(src=0).start()
    cap = cv2.VideoCapture(0)

    pass
except:
    cap = cv2.VideoCapture(0)


width = int(cap.get(3))  # float
height = int(cap.get(4))  # float
resultFrame = np.empty((height, width, 3), dtype=np.uint8)
resultFrame.fill(255)

while True:
    boolen, frame = cap.read()  # get the frame
    resultFrame = np.minimum(resultFrame, frame)
    cv2.imshow('final', resultFrame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.imwrite(args["output"], resultFrame)
cap.release()
cv2.destroyAllWindows()
