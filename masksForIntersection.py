import cv2 
import numpy as np
from numpy.testing._private.utils import integer_repr
from matplotlib import pyplot as plt

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
        [(0,height),(int(width/2.5),int(height/3.5)),(int(width/1.6),int(height/3.5)),(width,height)] #(y,x)
        ])
    elif option == 2: #options == 2 when the car turn right small
        polygons=np.array([
        [(int(width/1.65), int(height/1.68)),(width,int(height/4.1)),(width,height),(int(width/1.65),height)] #(y,x)
        ])
    elif option == 3:
        polygons=np.array([
        [(1,1),(650,350),(1030,350),(width,height)] #(y,x)
        ])

    # apply it to a block mask
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image
    
## to show the coordinates of an image:


def click_event(event,x,y,flags,params):
    #event -> left mouse click
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canny_image, str(x) + ',' + str(y), (x,y), font,1,(255,0,0), 2)
        cv2.imshow('image', canny_image)


# HOUGH LINES -> approximate presence of lines in an image
def houghLines(cropped_canny):
    return cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=5)

#making the points
def make_points(image, line):
    slope,intercept = line
    y1 = int(image.shape[0])
    y2 = int(y1*3.0/5)
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return [[x1,y1,x2,y2]]

#average line -> be more accurate 
def averaged_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        for x1,y1,x2,y2 in line:
            fit = np.polyfit((x1,x2), (y1,y2),1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                left_fit.append((slope,intercept))
            if slope >0: 
                right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_lines = make_points(image, left_fit_average)
    right_lines = make_points(image, right_fit_average)
    averaged_lines = [left_lines, right_lines]
    return averaged_lines

#dispaly the lines
def display_lines(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_img , (x1,y1), (x2,y2), (0,0,255), 10)
    return line_img

#to make the line be more visible 
def addWeighted(frame, line_img):
    return cv2.addWeighted(frame, 0.8, line_img, 1,1)


# for reviewing the mask

""" 
image = cv2.imread("Lanes.png")
lane_image = np.copy(image)
canny_image = canny(lane_image)
cv2.imshow("result", region_of_interest(canny_image,2))
cv2.waitKey(0)  
cv2.destroyAllWindows()
"""
# for finding the coordinates of an image
""" 
cv2.imshow('image', canny_image)
cv2.setMouseCallback('image', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
 """

# for the video

cap = cv2.VideoCapture("straight_simulation.mp4")
while (cap.isOpened()):
    _,frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image,1)
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
