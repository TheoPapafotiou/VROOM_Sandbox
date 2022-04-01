import numpy as np
import cv2
import time
from pedestrian_detection_jetson import PedestrianDetectionJetson

pedDet = PedestrianDetectionJetson()
video = cv2.VideoCapture('picam_pedestrian1.mp4')

time.sleep(2)

count = 0
try:
    while count < 2:
        color_image = cv2.imread('Pedestriandoll.jpeg')
        # _, color_image = video.read()

        start = time.time()
        result_image = pedDet.detectPedestrian(color_image)
        print("Time: ", time.time() - start)
        print()

        if count > 150:
            cv2.imwrite('RealSense_Ped_Detect_' + str(count) + '.jpg', result_image)

        count += 1
        time.sleep(0.1)

except Exception as e:
    print(e)