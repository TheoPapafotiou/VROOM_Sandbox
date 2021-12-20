import cv2
from matplotlib import image 
import matplotlib.pyplot as pl
import numpy as np 
import math

class CenterOfRoundanout():
    def __init__(self, image):
        self.image = image 
        self.width = image.shape[0]
        self.height = image.shape[1]
        self.thresh = None
        self.contours = None
        self.centers = None
        self.finalCenter = None

    
    def preprocessing(self):
        # BGR to GRAY
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Converting image to binary a binary one 
        _, binary = cv2.threshold(self.image, 254, 255, cv2.THRESH_BINARY_INV)
        # Setting a threshold to detect only clear lines 
        ret, thresh = cv2.threshold(binary, 240, 255, 0)
        self.thresh = thresh

    
    def findAndDrawContours(self):
    
        # find the contours from the thresholded image
        contours, hierarchy = cv2.findContours(self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # draw all contours
        self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        self.image = cv2.drawContours(self.image, contours, -1, (255, 0, 0), 1)
        self.contours = contours


    def get_contour_centers(self): 
    
        # ((x, y), radius) = cv2.minEnclosingCircle(c)
        centers = np.zeros((len(self.contours), 2), dtype=np.int16)
        for i, c in enumerate(self.contours):
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            centers[i] = center
        self.centers = centers
    

    def drawPoints(self):    

         #points =  get_contour_centers(contours)    
        for center in self.centers:
             self.image = cv2.circle(self.image, center, radius=0, color=(0, 0, 255), thickness= 5)    
            


    def findTheCenter(self):
        self.finalCenter = self.centers[0]
      
        closestCenter = 0
        x = self.centers[0][0]
        y = self.centers[0][1]
        smallestDistance = math.sqrt(abs((self.height - y) ** 2 + (0 - x) ** 2))

        for index, center in enumerate(self.centers[1:]):
            x = center[0]
            y = center[1]

            distance = math.sqrt((self.height - y) ** 2 + (0 - x) ** 2) #abs((y  - self.height) + x)

            print("Number {}: x is {}, y is {} and the distance is {}".format(index, x, y, distance))    
            
            if distance < smallestDistance:
                closestCenter = index + 1
                smallestDistance = distance

        self.image = cv2.circle(self.image, self.centers[closestCenter], radius=0, color=(20, 255, 80), thickness= 7)        


    def centerOfRoundabout(self):
        self.preprocessing()
        self.findAndDrawContours()
        self.get_contour_centers()
        self.drawPoints()
        self.findTheCenter()
        # Show the image. Print any button to quit
        
        pl.imshow(self.image)
        pl.show()
        cv2.waitKey(0)        

    def incompleteCircle(self):
        out = self.image.copy()
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        msk = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([15, 15, 15]))
        crc = cv2.HoughCircles(msk, cv2.HOUGH_GRADIENT, 0.5, 0.1, param1=50, param2=25, minRadius=0, maxRadius=0)

        # Ensure circles were found
        if crc is not None:
           crc = np.round(crc[0, :]).astype("int")

        # For each (x, y) coordinates and radius of the circles
           for (x, y, r) in crc:
               cv2.circle(out, (x, y), r, (0, 255, 0), 4)
               print("x:{}, y:{}".format(x, y))
        
        cv2.imshow("out", out)
        cv2.waitKey(0)




image = CenterOfRoundanout(cv2.imread('frame.png'))     
image.centerOfRoundabout()