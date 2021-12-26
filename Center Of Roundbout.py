import cv2
from matplotlib import image 
import matplotlib.pyplot as pl
import numpy as np 
import math
import sys
np.set_printoptions(threshold=sys.maxsize)

# Initialazing some colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (20, 255, 80)
BLUE = (0, 0, 255)


class CenterOfRoundanout():
    def __init__(self, image):
        self.image = image 
        self.image_copy = None
        self.width = image.shape[0]
        self.height = image.shape[1]
        self.thresh = None
        self.contours = None
        self.centers = None
        self.finalCenter = None
        self.closestCenter = None

    
    def preprocessing(self):
        """Saving an image copy and making the proper preprocessing"""
        # Saving a copy and converting it to RGB
        self.image_copy = self.image.copy()
        self.image_copy = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        # BGR to GRAY
        
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
         # Converting image to binary a binary one 
        _, binary = cv2.threshold(self.image, 254, 255, cv2.THRESH_BINARY_INV)
        # Setting a threshold to detect only clear lines 
        ret, thresh = cv2.threshold(binary, 240, 255, 0)
        self.thresh = thresh

    
    def findAndDrawContours(self):
        """Finding all the kind of contrours """
        # find the contours from the thresholded image
        contours, hierarchy = cv2.findContours(self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # draw all contours
        self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        self.image = cv2.drawContours(self.image, contours, -1, RED, 1)
        self.contours = contours


    def get_contour_centers(self): 
        """Finding all the possible centers of all possible contours"""
    
        # ((x, y), radius) = cv2.minEnclosingCircle(c)
        centers = np.zeros((len(self.contours), 2), dtype=np.int16)
        for i, c in enumerate(self.contours):
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            centers[i] = center
        self.centers = centers
    

    def drawPoints(self):    
        """Drawing all the possible center with small dots"""
             
        for center in self.centers:
             self.image = cv2.circle(self.image, center, radius=0, color=BLUE, thickness= 5)    
            

    def findTheCenter(self):
        """Finding the closest center to the left bottom point of the image"""
        self.finalCenter = self.centers[0]
      
        self.closestCenter = 0
        x = self.centers[0][0]
        y = self.centers[0][1]
        smallestDistance = math.sqrt(abs((self.height - y) ** 2 + (0 - x) ** 2))

        for index, center in enumerate(self.centers[1:]):
            x = center[0]
            y = center[1]

            distance = math.sqrt((self.height - y) ** 2 + (0 - x) ** 2)     
            
            if distance < smallestDistance:
                self.closestCenter = index + 1
                smallestDistance = distance

        self.image = cv2.circle(self.image, self.centers[self.closestCenter], radius=0, color=GREEN, thickness= 7)       

    def findTheLine(self):
          """Find the point that we will connect our created line"""

          if self.closestCenter:
              
              # Save the coordinates of the closest center
              x_closest = self.centers[self.closestCenter][0]
              y_closest = self.centers[self.closestCenter][1]         
                 
              for x in range(x_closest, self.image_copy.shape[1] - 1):
                  value = [self.image_copy[y_closest][x]]
                  if value[0][0] > 200 and value [0][1] > 200 and value [0][2] > 200:
                      self.image = cv2.circle(self.image, [x, y_closest], radius=0, color=GREEN, thickness= 8)  
                      cv2.line(self.image_copy, (0, 794), (x, y_closest), WHITE, thickness=7) ### Here is hardcoded. Needs to be fixed
                      break

    
    def findTheAngle(self):
        # Save the coordinates of the closest center
        x_closest = self.centers[self.closestCenter][0]
        y_closest = self.centers[self.closestCenter][1]
        y =  794 - y_closest
        x = x_closest

        AB = y
        BC = x
        AC = math.sqrt(AB ** 2 + BC **2)
        print("The angle is {}".format(math.degrees(math.atan(BC / AB)))) 



    def centerOfRoundabout(self):
        self.preprocessing()
        self.findAndDrawContours()
        self.get_contour_centers()
        self.drawPoints()
        self.findTheCenter()
        self.findTheLine()
        self.findTheAngle()
        
        # Show the image. Print any button to quit
        pl.imshow(self.image)
        pl.show()
        cv2.waitKey(0)        
  

image = CenterOfRoundanout(cv2.imread('frame3.png'))     
image.centerOfRoundabout()