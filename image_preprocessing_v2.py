from cgi import test
from re import S
import cv2
import time
import numpy as np
from LaneKeepingReloaded import LaneKeepingReloaded



class Roundabout:
    def __init__(self, image, option) -> None:
        self.image = image
        self.option = option

    def canny(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        # reduce noise  -> filter with guassian filter
        blur = cv2.GaussianBlur(gray,(5,5),0)
        #canny method to find lanes -> based on colour
        canny = cv2.Canny(blur, 50, 150)
        return canny

    def region_of_interest(self, canny_image):
        image = canny_image
        option = self.option
        #mask
        height = image.shape[0]
        width = image.shape[1]
        rect = None
        croped = None
        #for options == 1 when the car moves straight
        if option == 1:
            polygons=np.array([
            [(0,height),(int(width/1.5),int(height/2)),(int(width/1.5),int(height/1)),(width,height)] #(y,x)
            ])
        elif option == 2: #options == 2 when the car turn right small
            polygons=np.array([
            [(int(width/1.5), int(height/4.1)),(width,int(height/4.1)),(width,int(height/1.47)),(int(width/1.5),int(height/1.47))] #(y,x)
            ])
            rect = cv2.boundingRect(polygons)
            x,y,w,h = rect
            croped = image[y:y+h, x:x+w].copy()

        elif option == 3:
            polygons=np.array([
            [(1,1),(650,350),(1030,350),(width,height)] #(y,x)
            ])

        # apply it to a block mask
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, 255)

        masked_image = cv2.bitwise_and(image,mask)
        return masked_image, croped
    

    def roundaboutEntrance(self):

        canny_image = self.canny()
        cropped_image, smaller_image = self.region_of_interest(canny_image, 2)
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
       
        # Put side to side the real image and the image made above 
        numpy_horizontal_concat = np.concatenate((self.image, cropped_image), axis=1)
        cv2.imshow('frame', numpy_horizontal_concat)

        # Convert it back to gray
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
      
        # Tried to put 3 images side by a side
        test_image = np.concatenate((np.zeros(cropped_image.shape),cropped_image), axis=1)
        test_image = np.concatenate((test_image,np.zeros([cropped_image.shape[0],10]),test_image), axis=1)
        
        # Convert the image with the right way
        cropped_image[cropped_image.shape[0] - smaller_image.shape[0]: cropped_image.shape[0], 0: smaller_image.shape[1]] = smaller_image
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR) 

        # Lane keeping
        image = LaneKeepingReloaded(width=cropped_image.shape[0], height=cropped_image.shape[1])
        lines = image.lane_keeping_pipeline(cropped_image)
        cv2.imshow("Lane Keeping", lines[1])
        angle = lines[0]

        return angle



