import cv2 
import numpy as np
from numpy.testing._private.utils import integer_repr
from matplotlib import pyplot as plt

class Mask_intesecrtion:
    """
    THIS CLASS CREATES THE MASKS NEEDED FOR THE INTERSECTION
    """

    def canny(image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # reduce noise  -> filter with guassian filter
        blur = cv2.GaussianBlur(gray,(5,5),0)
        #canny method to find lanes -> based on colour
        canny = cv2.Canny(blur, 50, 150)
        return canny

    def region_of_interest(image,option):
        #mask
        height = image.shape[0]
        width = image.shape[1]
        #for options == 1 when the car moves straight
        if option == 1:
            polygons=np.array([
            [(0,height),(int(width/2.5),int(height*0.4)),(int(width/1.6),int(height*0.4)),(width,height)] #(y,x)
            ])
        elif option == 2: #options == 2 when the car turn right small
            polygons=np.array([
            [(int(width/1.45), int(height*0.5)),(width,int(height*0.5)),(width,height),(int(width/3.1),height)] #(y,x)
            ])
        elif option == 3: #options == 3 when the car turns big left
            polygons=np.array([
            [(1,1),(650,350),(1030,350),(width,height)] #(y,x)
            ])
        elif option == 4: # to detect the corner in intersection
            polygons=np.array([
            [(0,int(height/1.2)),(0,int(height/5)),(int(width/1.4),int(height/5)),(int(width),int(height/1.2))] #(y,x)
            ])    
        elif option == 5: # straight half cut
            polygons=np.array([
                [(0,int(height)),(int(width/2.5),int(height*0.4)),(int(width/2),int(height*0.4)),(int(width/2),int(height))] #(y,x)
                ])
        elif option == 6:
            return image
        # apply it to a block mask
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, 255)
        masked_image = cv2.bitwise_and(image,mask)
        return masked_image    
    
    #with the old camera the needed masks:

    def region_of_interest_old(image,option):
        #mask
        height = image.shape[0]
        width = image.shape[1]
        #for options == 1 when the car moves straight
        if option == 1:
            polygons=np.array([
            [(0,height),(int(width/2.5),int(height/3.5)),(int(width/1.6),int(height/3.5)),(width,height)] #(y,x)
            ])
        elif option == 2: #options == 2 when the car turn right small
            polygons=np.array([
            [(int(width/1.45), int(height/3)),(width,int(height/4.1)),(width,height),(int(width/3.1),height)] #(y,x)
            ])
        elif option == 3: #options == 3 when the car turns big left
            polygons=np.array([
            [(1,1),(650,350),(1030,350),(width,height)] #(y,x)
            ])
        elif option == 4: # to detect the corner in intersection
            polygons=np.array([
            [(0,int(height/1.2)),(0,int(height/5)),(int(width/1.4),int(height/5)),(int(width),int(height/1.2))] #(y,x)
            ])    
        elif option == 5: # straight half cut
            polygons=np.array([
                [(0,int(height)),(int(width/2.5),int(height/3.5)),(int(width/2),int(height/3.5)),(int(width/2),int(height))] #(y,x)
                ])
        elif option == 6:
            return image
        # apply it to a block mask
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, 255)
        masked_image = cv2.bitwise_and(image,mask)
        return masked_image
        
    ## to show the coordinates of an image:'''



    def click_event(event,x,y,canny_image):
        #event -> left mouse click
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x,' ', y)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(canny_image, str(x) + ',' + str(y), (x,y), font,1,(255,0,0), 2)
            cv2.imshow('image', canny_image)

    # for reviewing the mask

    # image = cv2.imread("photo/straight_1.png")
    # lane_image = np.copy(image)
    # canny_image = canny(lane_image)
    # cv2.imshow("result", region_of_interest(canny_image,5))
    # cv2.waitKey(0)  
    # cv2.destroyAllWindows()

    # for finding the coordinates of an image

    # cv2.imshow('image', canny_image)
    # cv2.setMouseCallback('image', click_event)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # for the video
"""
    cap = cv2.VideoCapture("videos/straight_simulation.mp4")
    while (cap.isOpened()):
        _,frame = cap.read()
        canny_image = canny(frame)
        cropped_image = region_of_interest(canny_image,4)
        #cv2.imshow("result", cropped_image)
        width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`  
        # we iterate through each corner, 
        # making a circle at each point that we think is a corner.
        cv2.imshow("result", cropped_image)

        if cv2.waitKey(5) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
    """


"""
def slope(x1,y1,x2,y2):
    if x2!=x1:
        return((y2-y1)/(x2-x1))
    else:
        return 'NA'

def drawLine(img, x1,y1,x2,y2):
    m = slope(x1,y1,x2,y2)
    h,w = img.shape[: 2]
    if m != 'NA':
        # starting point
        px = 0
        py = -(x1-0)*m + y1
        # ending point
        qx = w
        qy = -(x2-w)*m + y2
        cv2.line(img,(int(px), int(py)), (int(qx),int(qy)), (255,255,255), 12)
"""    


"""min_distance= mindist(img,width,0) # max initialization 
                        for i in range (1, self.counter2):
                            if min_distance > mindist(img,self.x_points[i],self.y_points[i]):
                                xmin = self.x_points[i]
                                ymin = self.y_points[i]
                                
                                
    def min_dist(self,img,x_point,y_point):
        width  = img.shape[0] 
        return ((((x_point - int(width/2) )**2) + ((y_point)**2) )**0.5)


"""
   

