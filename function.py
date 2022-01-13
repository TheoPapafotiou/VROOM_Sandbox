import cv2 
import numpy as np
import matplotlib.pyplot as plt
from numpy.testing._private.utils import integer_repr

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # reduce noise  -> filter with guassian filter
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #canny method to find lanes -> based on colour
    canny = cv2.Canny(blur, 50, 150)
    return canny

def click_event(event,x,y,flags,params):
    #event -> left mouse click
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canny_image, str(x) + ',' + str(y), (x,y), font,1,(255,0,0), 2)
        cv2.imshow('image', canny_image)


image = cv2.imread("slope_straight.png")
lane_image = np.copy(image)
canny_image = canny(lane_image)
canny = canny(lane_image)

cv2.imshow('image', canny_image)
cv2.setMouseCallback('image', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
