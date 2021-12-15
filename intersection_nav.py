import cv2 
import numpy as np
import matplotlib.pyplot as plt

def average_slope_intercept(image, lines):
    left_fit = []
    rigth_fit = []
    for line in lines :
        x1,y1,x2,y2 = line.reshape(4) # points of the line
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        print(parameters)





def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # reduce noise  -> filter with guassian filter
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #canny method to find lanes -> based on colour
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    #mask
    heigth = image.shape[0]
    polygons = np.array([
    [(0,heigth),(191,200),(500,200) ,(643, heigth)]
    ])
    # apply it to a block mask
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 225)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image 

def display_lines(image, lines):
    line_image  = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(225,0,0), 10)
    return line_image

image = cv2.imread('Lanes.png')
#convert it to black and white -> 0 - 250 -> 1 chanel so each faster than using 3 colours pixel 
lane_image = np.copy(image)
canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image)

# now we will use the hough transfrom technique to detect straight lines 
#HOUGH SPACE
# using r = x cos(u) + y sin(u)

lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
averaged_lines = average_slope_intercept(lane_image, lines)
line_image = display_lines(lane_image, lines)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1,1)
cv2.imshow("result", combo_image)
cv2.waitKey(0)

# using matplotlib to better show how to isolate this region
# pairno tin eikona se x,y axes

# canny = canny(lane_image)
# plt.imshow(canny)
# plt.show()

# stopped at  1:03:59
#https://www.youtube.com/watch?v=eLTLtUVuuy4&t=1488s&ab_channel=ProgrammingKnowledge
