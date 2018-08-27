from imutils import paths
import numpy as np
import imutils
import cv2
from imutils.video import VideoStream
import time
import argparse
from collections import deque

vs = VideoStream(src=0).start()

# allow the camera or video file to warm up
time.sleep(2.0)

i=0
while 1:
    i++
	frame = vs.read()
    cv2.imwrite("../images/cushion{}.jpg".format(i), resultFrame)

    key = cv2.waitKey(1) & 0xFF
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break
