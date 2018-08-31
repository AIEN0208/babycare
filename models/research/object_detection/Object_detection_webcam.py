######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/20/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a webcam feed.
# It draws boxes and scores around the objects of interest in each frame from
# the webcam.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.


# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import requests
import imutils
import dlib
import datetime
import collections
import time
import argparse
from imutils.video import VideoStream
from scipy.spatial import distance as dist
from imutils import face_utils
from keras.models import load_model
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import *

line_bot_api = LineBotApi('pTY9J+buz1s5ll/W5am0X8/Pm0s1NG1q5zp7etGVxFJMlNvuCC3BEatzG1vYSv4pIpyyO8y3CJKwa8CBe/7xIbneIJSOPM4gzt29ptfdakki13LhBaqzuy+EVcR6f2LMFGGyS5zV2RtO9AmwCi/3TwdB04t89/1O/w1cDnyilFU=')


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_data'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 2
urls = 'https://5d7907f6.ngrok.io/'
def photo_send(imagename):   # send photo to website server
	url = urls + 'api/covered/'
	datas = {'UserId': 1, 'ImageName' : imagename}
	files = {'Image': open('covered/' + imagename + '.jpg', 'rb')}
	response = requests.post(url, datas, files = files)
	line_bot_api.multicast(['U2311c55d2df0337059d20c7bce8fdc9f', 'U493e1e22128374c5acbecd583eb411fa', 'Uefbad5d2ac551bab8ad9329c51d97929','U7af0a3144cd9250d245004b1b2b34e70'], ImageSendMessage(
		original_content_url=urls + 'media/covered/'+imagename+'.jpg',
		preview_image_url=urls + 'media/covered/'+imagename+'.jpg'
	))
	

def send_alarm():   # send messages
	pt = datetime.datetime.now().strftime("%H:%M:%S")
	line_bot_api.multicast(['U2311c55d2df0337059d20c7bce8fdc9f', 'U493e1e22128374c5acbecd583eb411fa', 'Uefbad5d2ac551bab8ad9329c51d97929','U7af0a3144cd9250d245004b1b2b34e70'], TextSendMessage(text='Baby is waking up at ' + pt + '.'))

	


def nose_alarm():
	print("Baby's nose is obstructed!!")
	line_bot_api.multicast(['U2311c55d2df0337059d20c7bce8fdc9f', 'U493e1e22128374c5acbecd583eb411fa', 'Uefbad5d2ac551bab8ad9329c51d97929','U7af0a3144cd9250d245004b1b2b34e70'], TextSendMessage(text="Notice!! Baby's nose is obstructed!!"))


def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / C
	return ear

EYE_AR_THRESH = 0.4
EYE_AR_CONSEC_FRAMES = 20
NOSE_AR_CONSEC_FRAMES = 20

EYE_COUNTER = 0
NOSE_COUNTER = 0
capturecount = 0
n = 0

EYE_ALARMED = False
NOSE_ALARMED = False

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')


# Initialize webcam feed
# video = cv2.VideoCapture(0)
print("[INFO] starting video stream thread...")

video = VideoStream(0).start()
time.sleep(2.0)
# ret = video.set(3,1280)
# ret = video.set(4,720)

while(True):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    # ret, frame = video.read()
	frame = video.read()
	frame = imutils.resize(frame, width=960)
	oriframe = frame
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 1)

	frame_expanded = np.expand_dims(frame, axis=0)

	# Perform the actual detection by running the model with the image as input
	(boxes, scores, classes, num) = sess.run(
		[detection_boxes, detection_scores, detection_classes, num_detections],
		feed_dict={image_tensor: frame_expanded})

	# Draw the results of the detection (aka 'visulaize the results')
	
	if rects:
		vis_util.visualize_boxes_and_labels_on_image_array(
		frame,
		np.squeeze(boxes),
		'Aqua',
		np.squeeze(classes).astype(np.int32),
		np.squeeze(scores),
		category_index,
		use_normalized_coordinates=True,
		line_thickness=8,
		min_score_thresh=0.60)
		for (i, rect) in enumerate(rects):
			n = 1
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			nose = shape[nStart:nEnd]

			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)

			ear = (leftEAR + rightEAR) / 2.0

			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)


			if ear > EYE_AR_THRESH:
				EYE_COUNTER += 1

				if EYE_COUNTER >= EYE_AR_CONSEC_FRAMES:
					if not EYE_ALARMED:
						send_alarm()
						EYE_ALARMED = True
						capturecount = 20
			else:
				EYE_COUNTER = 0
				ALARM_ON = False

			if capturecount > 0 :
				cv2.putText(frame, "child wakeup!!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				capturecount -= 1

			NOSE_COUNTER = 0
			frame = frame.copy()

			# put nose dots
			for (x, y) in nose:
				cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

			# put face index
			(x, y, w, h) = face_utils.rect_to_bb(rect)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 255), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 255), 1)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.putText(frame, "EAR: {:.0f}".format(ear*100), (800, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	elif n != 0:
		NOSE_COUNTER += 1

		vis_util.visualize_boxes_and_labels_on_image_array(
		frame,
		np.squeeze(boxes),
		'Blue',
		np.squeeze(classes).astype(np.int32),
		np.squeeze(scores),
		category_index,
		use_normalized_coordinates=True,
		line_thickness=8,
		min_score_thresh=0.60)

		cv2.putText(frame, "Notice!! Nose be covered!!", (10, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		if NOSE_COUNTER >= NOSE_AR_CONSEC_FRAMES:
			if NOSE_ALARMED == False:
				nose_alarm()
				now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
				save_path = 'covered/'+now+'.jpg'
				cv2.imwrite(save_path, frame)
				print("success capture")
				photo_send(now)
				
				NOSE_ALARMED = True
				

	# All the results have been drawn on the frame, so it's time to display it.
	cv2.imshow('Object detector', frame)
	key = cv2.waitKey(1) & 0xFF
	
	# Press 'q' to quit
	if key == ord('q'):
		break
	if key == ord("n"):
		NOSE_COUNTER = 0
		NOSE_ALARMED = False
		EYE_COUNTER = 0
		EYE_ALARMED = False
# Clean up
cv2.destroyAllWindows()
video.stop()

