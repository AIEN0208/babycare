import os 
import cv2
 
def blue():
    os.chdir("/home/aien04/baobaoBot/models/research/object_detection")
    os.system("python3 Object_detection_webcam.py")

def black():
    os.chdir("/home/aien04/baobaoBot/soundtest")
    os.system("python3 soundtest.py")

def green():
    os.chdir("/home/aien04/baobaoBot/facere")
    os.system("python3 main.py")

def yello():
    os.chdir("/home/aien04/baobaoBot/happy_face_0816")
    os.system("python3 happy_face.py")

def red():
    os.chdir("/home/aien04/baobaoBot/BOAT")
    os.system("python3 sketching.py")

mode1,mode2 = 0,0
# print("Press 1 for mode1, 2 for mode2.")
while 1:
   
    key = input("Press 1 for mode1, 2 for mode2.")
    if key == "1":
        mode1 = 1
        break
    if key == "2":
        mode2 = 1
        break


while mode1:
    key = input("Press 1,2,3,4,5 for ...........")
    if key == '1':
        blue()

    if key == '2':
        black()

    if key == '3':
        green()

    if key == '4':
        yello()

    if key == '5':
        red()

    if key == 'q':
        break

while mode2:
    key = input("Press q for next one")
    if key == 'q':
        blue()
        key = input("Press q for next one")
        if key ==  'q':
            black()
            key = input("Press q for next one")
            if key ==  'q':
                green()
                key = input("Press q for next one")
                if key ==  'q':
                    yello()
                    key = input("Press q for next one")
                    if key ==  'q':
                        red()
                        key = input("Press q for next one")
                        if key ==  'q':
                            print('over')
                            break
                        
