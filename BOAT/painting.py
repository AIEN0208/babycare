# import the necessary packages
import argparse
import cv2
import numpy as np
from imutils.video import VideoStream
import os


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video", required=True, help="path to input video file")
#ap.add_argument("-o", "--output", required=True, help="path to output png")
args = vars(ap.parse_args())
# try:
#     # cap = VideoStream(src=0).start()
#     cap = cv2.VideoCapture(0)

#     pass
# except:
cap = cv2.VideoCapture(1)
if cap.read()[1] is None:
    cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

redLower = (0, 200, 50)
redUpper = (50, 255, 255)

yellowLower = (0, 200, 200)
yellowUpper = (50, 255, 255)

greenLower = (50, 150, 50)
greenUpper = (100, 255, 100)

blueLower = (100, 50, 50)
blueUpper = (255, 150, 150)

blackLower = (0, 0, 0)
blackUpper = (150, 150, 150)

# greenLower = (0, 0, 0)
# greenUpper = (255, 255, 255)

width = int(cap.get(3))  # float
height = int(cap.get(4))  # float
resultFrame = np.empty((height, width, 3), dtype=np.uint8)
res0 = resultFrame
resultFrame.fill(255)

while True:
    boolen, frame = cap.read()  # get the frame

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (1, 1), 0)
    # # gray = cv2.erode(gray, None, iterations=2)

    # edged = cv2.Canny(gray, 35, 125)
    # # edged = cv2.dilate(edged, None, iterations=1)
    # edged = cv2.bitwise_not(edged)
    # resultFrame = np.minimum(resultFrame, edged)
    # frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    redmask = cv2.inRange(hsv, redLower, redUpper)
    redmask = cv2.erode(redmask, None, iterations=2)
    redmask = cv2.dilate(redmask, None, iterations=2)

    yellowmask = cv2.inRange(hsv, yellowLower, yellowUpper)
    yellowmask = cv2.erode(yellowmask, None, iterations=2)
    yellowmask = cv2.dilate(yellowmask, None, iterations=2)

    greenmask = cv2.inRange(hsv, greenLower, greenUpper)
    greenmask = cv2.erode(greenmask, None, iterations=2)
    greenmask = cv2.dilate(greenmask, None, iterations=2)

    bluemask = cv2.inRange(hsv, blueLower, blueUpper)
    bluemask = cv2.erode(bluemask, None, iterations=2)
    bluemask = cv2.dilate(bluemask, None, iterations=2)

    blackmask = cv2.inRange(hsv, blackLower, blackUpper)
    # blackmask = cv2.erode(blackmask, None, iterations=5)
    blackmask = cv2.dilate(blackmask, None, iterations=1)
    blackmask = cv2.GaussianBlur(blackmask, (5, 5), 0)
    blackmask = cv2.Canny(blackmask, 35, 125)


    # find target
    redmasked = cv2.bitwise_and(frame, frame, mask=redmask)
    yellowmasked = cv2.bitwise_and(frame, frame, mask=yellowmask)
    greenmasked = cv2.bitwise_and(frame, frame, mask=greenmask)
    bluemasked = cv2.bitwise_and(frame, frame, mask=bluemask)
    blackmasked = cv2.bitwise_and(frame, frame, mask=blackmask)
    # locates the area
    redmask = cv2.bitwise_not(redmask)
    yellowmask = cv2.bitwise_not(yellowmask)
    greenmask = cv2.bitwise_not(greenmask)
    bluemask = cv2.bitwise_not(bluemask)
    blackmask = cv2.bitwise_not(blackmask)
    # dig hole
    rfakemask = cv2.bitwise_and(resultFrame, resultFrame, mask=redmask)
    resultFrame = np.minimum(resultFrame, rfakemask)
    yfakemask = cv2.bitwise_and(resultFrame, resultFrame, mask=yellowmask)
    resultFrame = np.minimum(resultFrame, yfakemask)
    gfakemask = cv2.bitwise_and(resultFrame, resultFrame, mask=greenmask)
    resultFrame = np.minimum(resultFrame, gfakemask)
    bfakemask = cv2.bitwise_and(resultFrame, resultFrame, mask=bluemask)
    resultFrame = np.minimum(resultFrame, bfakemask)
    # dfakemask = cv2.bitwise_and(resultFrame, resultFrame, mask=blackmask)
    # resultFrame = np.minimum(resultFrame, dfakemask)
    # fakemask = cv2.bitwise_and(frame, frame, mask=redmask)
    # put target on fakemask
    resultFrame = cv2.add(redmasked, resultFrame)
    resultFrame = cv2.add(yellowmasked, resultFrame)
    resultFrame = cv2.add(greenmasked, resultFrame)
    resultFrame = cv2.add(bluemasked, resultFrame)
    # resultFrame = cv2.add(blackmasked, resultFrame)
    # resultFrame = cv2.bitwise_and(frame,frame,mask=fakemask)
    # resultFrame = np.minimum(redmasked, fakemask)
    # resultFrame = np.minimum(yellowmasked, fakemask)
    # resultFrame = np.minimum(greenmasked, fakemask)
    # resultFrame = np.maximum(masked, fakemask2)
    # resf = cv2.flip(resultFrame, 1)
    resf = cv2.flip(resultFrame, 1)
    cv2.imshow('final', resf)
    # cv2.imshow('final', np.hstack([resf,redmasked,yellowmasked,greenmasked]))

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.imwrite('painting.jpg', resf)
cap.release()
cv2.destroyAllWindows()

def next():
    os.system("python3 artwork.py")
next()
