# USAGE
# python distance_to_camera.py

# import the necessary packages
from imutils import paths
import numpy as np
import imutils
import cv2
from imutils.video import VideoStream
import time
import argparse
from collections import deque


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# def find_marker(image):
# 	# convert the image to grayscale, blur it, and detect edges
# 	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 	gray = cv2.GaussianBlur(gray, (5, 5), 0)
# 	edged = cv2.Canny(gray, 35, 125)

# 	# find the contours in the edged image and keep the largest one;
# 	# we'll assume that this is our piece of paper in the image
# 	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# 	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
# 	c = max(cnts, key = cv2.contourArea)

# 	# compute the bounding box of the of the paper region and return it
# 	return cv2.minAreaRect(c)

def find_marker(image):
	# resize the frame, blur it, and convert it to the HSV
	# color space
	# frame = imutils.resize(image, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	global redmask
	redmask = cv2.inRange(hsv, redLower, redUpper)
	redmask = cv2.erode(redmask, None, iterations=2)
	redmask = cv2.dilate(redmask, None, iterations=2)

	# convert the image to grayscale, blur it, and detect edges
	# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# gray = cv2.GaussianBlur(gray, (5, 5), 0)
	# edged = cv2.Canny(gray, 35, 125)

	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	# cnts = cv2.findContours(redmask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	# cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	# c = max(cnts, key = cv2.contourArea)

	redcnts = cv2.findContours(redmask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	# print(redcnts)	
	redcnts = redcnts[1] #if imutils.is_cv2() else redcnts[1]
	if redcnts:
		rc = max(redcnts, key=cv2.contourArea)
		return cv2.minAreaRect(rc), rc
	else:
		return 0, 0

	# compute the bounding box of the of the paper region and return it

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

redLower = (0, 200, 50)
redUpper = (50, 255, 255)

pts = deque(maxlen=args["buffer"])
rpts = [None]
rthickness = [None]

# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 20.0

# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 5.0

# image = cv2.imread("images/2ft.png")
# marker = find_marker(image)
# focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

vs = VideoStream(src=0).start()
cap = cv2.VideoCapture(0)
WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(WIDTH,HEIGHT)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(WIDTH,HEIGHT)

# allow the camera or video file to warm up
time.sleep(2.0)


while 1:
	frame = vs.read()
	frame = imutils.resize(frame, width=600)
	marker,rc = find_marker(frame)
	if marker:
		focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
		inches0 = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
		break
	# else:
	# 	key = cv2.waitKey(1)
	# 	if key:
	# 		break

# keep looping
while True:

	# load the furst image that contains an object that is KNOWN TO BE 2 feet
	# from our camera, then find the paper marker in the image, and initialize
	# the focal length
	# image = cv2.imread("images/2ft.png")
	frame = vs.read()
	frame = imutils.resize(frame, width=1080)

	# frame = frame[1]
	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break
	# if not image: print("T")

	# load the image, find the marker in the image, then compute the
	# distance to the marker from the camera
	# image = cv2.imread(imagePath)
	marker, rc = find_marker(frame)
	if marker:
		inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
		# draw a bounding box around the image and display it
		box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
		box = np.int0(box)
		cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
		# cv2.putText(frame, "%.2fft" % (inches / 12),
		# 	(frame.shape[1] - 200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
		# 	2.0, (0, 255, 0), 3)

		((rx, ry), rradius) = cv2.minEnclosingCircle(rc)
		rM = cv2.moments(rc)
		redcenter = (int(rM["m10"] / rM["m00"]), int(rM["m01"] / rM["m00"]))
		
	if inches  < inches0 -10:
		if rradius > 5 and rradius < 300:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			# cv2.circle(frame, (int(rx), int(ry)), int(rradius),
			# 	(0, 0, 255), 2)
			cv2.circle(frame, redcenter, 5, (0, 0, 255), -1)
		# rpts.appendleft(redcenter)	
		rpts.append(redcenter)	
		rthickness.append(int( np.exp((inches0 - 0) / (inches)) * 1))
		# print(rpts)
		for i in range(1, len(rpts)):
			# if either of the tracked points are None, ignore
			# them
			if rpts[i - 1] is None or rpts[i] is None:
				continue

			# otherwise, compute the thickness of the line and
			# draw the connecting lines
			# rthickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
			# rthickness = int( (inches+inches0) / inches * 5)

			# cv2.line(frame, rpts[i - 1], rpts[i], (0, 0, 255), 5)
			cv2.line(frame, rpts[i - 1], rpts[i], (0, 0, 255), rthickness[i - 1])
	else:
		rpts[-1] = None
		rthickness[-1] = None
		if rradius > 5 and rradius < 100:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			# cv2.circle(frame, (int(rx), int(ry)), int(rradius),
			# 	(0, 0, 255), 2)
			cv2.circle(frame, redcenter, 5, (0, 255, 255), -1)
		pts.appendleft(redcenter)	
		for i in range(1, len(pts)):
			# if either of the tracked points are None, ignore
			# them
			if pts[i - 1] is None or pts[i] is None:
				continue

			# otherwise, compute the thickness of the line and
			# draw the connecting lines
			thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
			# rthickness = int(64 / float(i + 1)) * 2.5)

			# cv2.line(frame, rpts[i - 1], rpts[i], (0, 0, 255), 5)
			cv2.line(frame, pts[i - 1], pts[i], (0, 255, 255), thickness)
		xpos = [pos[0] for pos in pts if pos is not None]
		ypos = [pos[1] for pos in pts if pos is not None]
		masscnt = (sum(xpos)/args["buffer"],sum(ypos)/args["buffer"])
		dist = [((pos[0]-masscnt[0])**2+(pos[1]-masscnt[1])**2) for pos in pts if pos is not None]
		# print(xpos, ypos)
		# print(masscnt)
		# print(dist)
		#print((np.max(dist)/np.min(dist))**0.5)
		if (np.max(dist)/np.min(dist))**0.5 < 2 and np.max(dist)/np.min(dist) > 1.0 and np.max(dist)-np.min(dist) > 300:
			rpts = [None]
			rthickness = [None]
			#print('Clean')


	# show the frame to our screen
	frame = cv2.flip(frame, 1)
	redmask = cv2.flip(redmask, 1)
	masked = cv2.bitwise_and(frame, frame, mask=redmask)	
	# masked = cv2.flip(masked, 1)
	# cv2.imshow("image", frame)
	cv2.imshow("Frame", np.hstack([frame, masked]))
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()

# otherwise, release the camera
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()