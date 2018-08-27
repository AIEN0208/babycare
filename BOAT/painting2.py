import cv2
import numpy as np
from imutils import paths
import imutils
from imutils.video import VideoStream
import time
import argparse
from collections import deque
import os


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=24,
	help="max buffer size")
args = vars(ap.parse_args())

hand_hist = None
traverse_point = []
total_rectangle = 9
hand_rect_one_x = None
hand_rect_one_y = None

hand_rect_two_x = None
hand_rect_two_y = None

pts = deque(maxlen=args["buffer"])
rpts = [None]
rthickness = [None]
focalLength = None
hist_mask_image = None
#marker = None



# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 20.0

# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 5.0

def rescale_frame(frame, wpercent=130, hpercent=130):
    width = int(frame.shape[1] * wpercent / 100)
    height = int(frame.shape[0] * hpercent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def contours(hist_mask_image):
    gray_hist_mask_image = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_hist_mask_image, 0, 255, 0)
    _, cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#RETR_EXTERNAL,RETR_TREE
    return cont


def max_contour(contour_list):
    max_i = 0
    max_area = 0

    for i in range(len(contour_list)):
        cnt = contour_list[i]

        area_cnt = cv2.contourArea(cnt)

        if area_cnt > max_area:
            max_area = area_cnt
            max_i = i

        return contour_list[max_i]


def draw_rect(frame):
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    hand_rect_one_x = np.array(
        [19 * rows / 40, 19 * rows / 40, 19 * rows / 40, 20 * rows / 40, 20 * rows / 40, 20 * rows / 40, 21 * rows / 40,
         21 * rows / 40, 21 * rows / 40], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [19 * cols / 40, 20 * cols / 40, 21 * cols / 40, 19 * cols / 40, 20 * cols / 40, 21 * cols / 40, 19 * cols / 40,
         20 * cols / 40, 21 * cols / 40], dtype=np.uint32)

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(total_rectangle):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (50, 150, 200), 3)

    return frame


def hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

    for i in range(total_rectangle):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                          hand_rect_one_y[i]:hand_rect_one_y[i] + 10]

    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)


def hist_masking(frame, hist):
    global thresh
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, disc, dst)

    ret, thresh = cv2.threshold(dst, 200, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=20)
    thresh = cv2.dilate(thresh, None, iterations=5)

    threshm = cv2.merge((thresh, thresh, thresh))

    return cv2.bitwise_and(frame, threshm)


def centroid(max_contour):
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None


def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None


def draw_circles(frame, traverse_point):
    if traverse_point is not None:
        for i in range(len(traverse_point)):
            cv2.circle(frame, traverse_point[i], int(5 - (5 * i * 3) / 100), [0, 255, 255], -1)


def manage_image_opr(frame, hand_hist):
    global focalLength, hist_mask_image, rpts,rthickness, marker, rradius, redcenter, inches0, thresh, resultFrame
    hist_mask_image = hist_masking(frame, hand_hist)
    contour_list = contours(hist_mask_image)
    max_cont = max_contour(contour_list)

    cnt_centroid = centroid(max_cont)
    cv2.circle(frame, cnt_centroid, 5, [0, 255, 255], -1)

	#redcnts = cv2.findContours(hist_mask_image.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if cnt_centroid:
        marker = cv2.minAreaRect(max_cont)
    else:
        marker = 0
    if marker:
        inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
        # draw a bounding box around the image and display it
        box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
        box = np.int0(box)
        cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
        # cv2.putText(frame, "%.2fft" % (inches / 12),
        # 	(frame.shape[1] - 200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
        # 	2.0, (0, 255, 0), 3)

        ((rx, ry), rradius) = cv2.minEnclosingCircle(max_cont)
        # rx, ry = cnt_centroid
        rM = cv2.moments(max_cont)
        redcenter = (int(rM["m10"] / rM["m00"]), int(rM["m01"] / rM["m00"]))

        if inches  < inches0 -1:
            if 1:
                # find target
                redmasked = hist_mask_image
                # locates the area
                redmask = cv2.bitwise_not(thresh)
                # dig hole
                rfakemask = cv2.bitwise_and(resultFrame, resultFrame, mask=redmask)
                resultFrame = np.minimum(resultFrame, rfakemask)
                # put target on fakemask
                # ret, redmasked = cv2.threshold(redmasked, 0, 255, cv2.THRESH_BINARY)                
                # redmasked = cv2.medianBlur(redmasked, 17)
                # redmasked = cv2.bitwise_and(redmasked, redmasked, mask=thresh)
                # ret, redmasked = cv2.threshold(redmasked, 0, 255, cv2.THRESH_BINARY)                
                resultFrame = cv2.add(redmasked, resultFrame)
            if 1:#rradius > 5 and rradius < 300:
                cv2.circle(frame, redcenter, 5, (0, 0, 255), -1)	
            if 0:
                rpts.append(redcenter)	
                print(inches0, inches)
                rthickness.append(int( np.exp((inches0 - 0) / (inches)) * 1))
                # print(rpts)
                for i in range(1, len(rpts)):
                    # if either of the tracked points are None, ignore
                    # them
                    if rpts[i - 1] is None or rpts[i] is None:
                        continue
                    cv2.line(frame, rpts[i - 1], rpts[i], (0, 0, 255), rthickness[i - 1])
        else:
            rpts[-1] = None
            rthickness[-1] = None
            if 1:#rradius > 5 and rradius < 100:
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
            # print((np.max(dist)/np.min(dist))**0.5)
            # if (np.max(dist)/np.min(dist))**0.5 < 2 and np.max(dist)/np.min(dist) > 1.0 and np.max(dist) < 1000 and np.min(dist) > 100:#np.max(dist)-np.min(dist) < np.mean(dist)*0.3 and
            #     rpts = [None]
            #     rthickness = [None]
            #     resultFrame.fill(255)
            #     print(np.max(dist))
            #     print('Clean')

    if None: #max_cont is not None:
        hull = cv2.convexHull(max_cont, returnPoints=False)
        defects = cv2.convexityDefects(max_cont, hull)
        far_point = farthest_point(defects, max_cont, cnt_centroid)
        print("Centroid : " + str(cnt_centroid) + ", farthest Point : " + str(far_point))
        cv2.circle(frame, far_point, 5, [0, 0, 255], -1)
        if len(traverse_point) < 20:
            traverse_point.append(far_point)
        else:
            traverse_point.pop(0)
            traverse_point.append(far_point)

        draw_circles(frame, traverse_point)


def main():
    global hand_hist, focalLength, hist_mask_image, inches0, resultFrame
    is_hand_hist_created = False
    capture = cv2.VideoCapture(1)
    if capture.read()[1] is None:
        capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    width = int(capture.get(3))  # float
    height = int(capture.get(4))  # float
    resultFrame = np.empty((height, width, 3), dtype=np.uint8)
    res0 = resultFrame
    resultFrame.fill(255)
    resf = None

    while capture.isOpened():
        pressed_key = cv2.waitKey(1)
        _, frame = capture.read()
        frame = cv2.flip(frame,1)

        if pressed_key & 0xFF == ord('s'):
            is_hand_hist_created = True
            hand_hist = hand_histogram(frame)
            hist_mask_image = hist_masking(frame, hand_hist)
            contour_list = contours(hist_mask_image)
            max_cont = max_contour(contour_list)
            # print(max_cont, contour_list)
            cnt_centroid = centroid(max_cont)
            marker = cv2.minAreaRect(max_cont)
            focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
            inches0 = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
            print(marker[1][0],inches0)
            #cv2.destroyAllWindows()

        if is_hand_hist_created:
            manage_image_opr(frame, hand_hist)

        frame = draw_rect(frame)

        if hist_mask_image is not None and resf is not None:
            pass
            #cv2.imshow("Live Feed", np.hstack([frame,hist_mask_image,resf]))
        else:
            img = np.uint8(np.clip((1.3 * frame + 10), 0, 255))
            paint = np.uint8(np.clip((1.5 * resultFrame + 10), 0, 255))
            # if 1:#is_hand_hist_created:
            cv2.imshow('final', np.hstack([img,paint]))
            # else:
            #    cv2.imshow("Live Feed", img)




        if pressed_key & 0xFF == ord('q'):
            break
    cv2.imwrite('painting.jpg', paint)
    cv2.destroyAllWindows()
    capture.release()

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

if __name__ == '__main__':
    main()

def next():
    os.system("python3 artwork2.py")
next()



