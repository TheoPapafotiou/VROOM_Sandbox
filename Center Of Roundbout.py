import cv2
from matplotlib import image 
import matplotlib.pyplot as pl
import numpy as np 


class CenterOfRoundanout():
    def __init__(self, image):
        self.image = image 
        self.thresh = None
        self.contours = None
        self.centers = None

    
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


    def centerOfRoundabout(self):
        self.preprocessing()
        self.findAndDrawContours()
        self.get_contour_centers()
        self.drawPoints()
        # Show the image. Print any button to quit
        
        pl.imshow(self.image)
        pl.show()
        cv2.waitKey(0)
        
image = CenterOfRoundanout(cv2.imread('frame.png'))     
image.centerOfRoundabout()

