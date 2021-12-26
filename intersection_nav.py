import cv2 
import numpy as np
import matplotlib.pyplot as plt

# def translate_image(img, st_angle, low_x_range, high_x_range, low_y_range, high_y_range, delta_st_angle_per_px):
#     """
#     Shifts the image right, left, up or down. 
#     When performing a lateral shift, a delta proportional to the pixel shifts is added to the current steering angle 
#     """
#     rows, cols = (img.shape[0], img.shape[1])
#     translation_x = np.random.randint(low_x_range, high_x_range) 
#     translation_y = np.random.randint(low_y_range, high_y_range) 
    
#     st_angle += translation_x * delta_st_angle_per_px
#     translation_matrix = np.float32([[1, 0, translation_x],[0, 1, translation_y]])
#     img = cv2.warpAffine(img, translation_matrix, (cols, rows))
    
#     return img, st_angle

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
   # print(image.shape)
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = (int((y2-intercept)/slope))
    print(x1,'x1')
    print(x2,'x2')
    print(y1,'y1')
    print(y2,'y2')
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines :
        x1,y1,x2,y2 = line.reshape(4) # points of the line
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
    # if the line has slope < 0  then it is on the left (as the x increases the y decreases) 
    # otherwise on the right -> slope > 0
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    # now i will find the avergae of all lines
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # reduce noise  -> filter with guassian filter
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #canny method to find lanes -> based on colour
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image, lines):
    line_image  = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0), 10)
    return line_image

def region_of_interest(image):
    #mask
    heigth = image.shape[0]
#   polygons = np.array([
#     [(0,heigth),(191,200),(500,200) ,(643, heigth)]
#     ])  
    polygons=np.array([
        [(0,heigth),(65,110),(340,110),(400,heigth)] #(y,x)
        ])
    # apply it to a block mask
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image 

image = cv2.imread('Lanes.png')
#convert it to black and white -> 0 - 250 -> 1 chanel so each faster than using 3 colours pixel 
lane_image = np.copy(image)
canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image)

# now we will use the hough transfrom technique to detect straight lines 
#HOUGH SPACE
# using r = x cos(u) + y sin(u)

# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
# averaged_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, averaged_lines)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1,1)
# cv2.imshow("result", combo_image)
# cv2.waitKey(0)

# # using matplotlib to better show how to isolate this region
# pairno tin eikona se x,y axis

# canny = canny(lane_image)
# plt.imshow(canny)
# plt.show()

cap = cv2.VideoCapture("test2.mp4")
while (cap.isOpened()):
    _,frame = cap.read()
    # the algorithm to detect lines 
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
   # averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1,1)
    cv2.imshow("result", combo_image)
    if cv2.waitKey(1)  == ord('q'):
    #wait 1 milisec 
        break
cap.release()
cv2.destroyAllWindows()



