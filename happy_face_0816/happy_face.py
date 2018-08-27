from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import sys
import dlib
import cv2
import requests
import datetime
import tensorflow as tf
from keras.models import load_model

emotion_labels = ["cry", "happy", "neutral"]
model = load_model('./babyemotion_03.h5')
model.compile(optimizer=tf.train.AdamOptimizer(), loss='categorical_crossentropy')


def photo_send(imagename):   # send photo to website server
	url = 'http://13.229.127.168/api/album/'
	datas = {'UserId': 1, 'ImageName' : imagename}
	files = {'Image': open('images/' + imagename + '.jpg', 'rb')}
	response = requests.post(url, datas, files = files)
	print(response.text)


print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

happycount = 0
capturecount = 0
pic = ''

print("[INFO] starting video stream thread...")
vs = VideoStream(1).start()
time.sleep(2.0)

while True:

	frame = vs.read()
	frame = imutils.resize(frame, width=960)
	oriframe = frame
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	rects = detector(gray, 1)
	if rects:
		for (i, rect) in enumerate(rects):
			n = 1
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			# emotion
			nosecenter = shape[30]
			adshape = []
			for (x, y) in shape:
				adshape.append([x-nosecenter[0], y-nosecenter[1]])
			adshape = np.array(adshape)
			#adshape = (adshape - adshape.mean(axis=0))/adshape.std(axis=0)
			adshape = adshape.reshape(1, adshape.shape[0] * adshape.shape[1])
			pred = model.predict(adshape)
			emo = np.argmax(pred)
			emotion = emotion_labels[emo]
			
			# take picture
			if pred[0][1] > 0.5:
				happycount += 1
				if happycount > 10:
					now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
					if now[-8:-3] != pic or pic == '':
						save_path = 'images/'+now+'.jpg'
						cv2.imwrite(save_path, oriframe)
						print("success capture")
						photo_send(now)
						capturecount = 20
						happycount = 0
						pic = now[-8:-3]

			# capturecount
			if capturecount > 0 :
				cv2.putText(frame, "Success capture!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				capturecount -= 1

			# put face index
			(x, y, w, h) = face_utils.rect_to_bb(rect)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.putText(frame, emotion, (x+w-30, y-10), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break


cv2.destroyAllWindows()
vs.stop()
