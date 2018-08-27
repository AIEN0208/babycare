from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import cv2

img = cv2.imread("panda.png")
if img
print(img)
# cv2.imshow(img,img)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv)
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
for x, c, z in zip([h,s,v], ['r', 'g', 'b'], [30, 20, 10]):
    xs = np.arange(256)
    ys = cv2.calcHist([x], [0], None, [256], [0,256])
    cs = [c] * len(xs)
    cs[0] = 'c'
    ax.bar(xs, ys.ravel(), zs=z, zdir='y', color=cs, alpha=0.8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()