import numpy as np
import cv2
import time
from pedestrian_detection_jetson import PedestrianDetectionJetson

pedDet = PedestrianDetectionJetson() # ~ 6 secs delay due to initializations

count = 0
try:
    # Wait for a coherent pair of frames: depth and color
    color_image = cv2.imread('pose.png')

    start = time.time()
    result_image = pedDet.detectPedestrian()
    print("Time: ", time.time() - start)

    cv2.imwrite('RealSense_Ped_Detect_' + str(count) + '.jpg', color_image)

    time.sleep(0.2)

except Exception as e:
    print(e)