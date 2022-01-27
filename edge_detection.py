# edge detection 
from cgitb import grey
from re import X
from xml.etree.ElementPath import xpath_tokenizer
import cv2
import matplotlib.pyplot as plt 
from matplotlib import pyplot as plt
import numpy as np
from numpy.core.overrides import verify_matching_signatures
from masksForIntersection import Mask_intesecrtion
from skimage.measure import ransac, LineModelND
from datetime import datetime

import time 

x_points = []
y_points = []

def ransac_method(x_points, y_points):
    data = np.column_stack([x_points,y_points])

    model = LineModelND()
    model.estimate(data)
    # robustly fit line only using inlier data with RANSAC algorithm
    model_robust, inliers = ransac(data, LineModelND, min_samples=2,
                               residual_threshold=1, max_trials=1000)
    outliers = inliers == False
    line_x = x_points
    line_y = model.predict_y(line_x)
    line_y_robust = model_robust.predict_y(line_x)
    return line_x, line_y_robust

'''
    fig, ax = plt.subplots()
    ax.plot(data[inliers, 0], data[inliers, 1], '.b', alpha=0.6,
            label='Inlier data')
    ax.plot(data[outliers, 0], data[outliers, 1], '.r', alpha=0.6,
            label='Outlier data')
    ax.plot(line_x, line_y, '-k', label='Line model from all data')
    ax.plot(line_x, line_y_robust, '-b', label='Robust line model')
    ax.legend(loc='lower left')
    #plt.show()
 '''
    




def corner_3(gray,turn,counter): #turn = 1 : straight, turn = 2 : small right, turn = 3 : left big
    dst = cv2.cornerHarris(gray, 2,3,0.04)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)
    
# find centroids
    ret,lables, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS +cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    res = np.hstack((centroids,corners))
    res = np.int0(res)
   

    for i in res:
        x,y,x2,y2 = i.ravel() 
        if turn == 1:
            x_points.append(x)
            y_points.append(y)            
            cv2.circle(gray, (x, y), 3, 255, -1)
        elif turn == 2:
            if x == x2 and y == y2:
#                data = np.column_stack([x,y])
                x_points.append(x)
                y_points.append(y) 
               # data = np.column_stack([x_points,y_points])           
                cv2.circle(gray, (x, y), 3, 255, -1)

    xnew,ynew = ransac_method(x_points,y_points)
    
    white = (255,255,255)
   
    if len(x_points) > 2:
        cv2.line(gray, (0, int(height)),(int(xnew[0]), int(ynew[0])), white, 9) 
    '''
    if len(x_points) > 2:
        cv2.line(gray, (0, int(height)),(x_points[1], y_points[1]), white, 9)
    '''
    return gray


    
cap = cv2.VideoCapture("videos/straight.mp4")
counter = 0
while (cap.isOpened()):
    counter = counter +  1
    _,frame = cap.read()
    gray = Mask_intesecrtion.canny(frame)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    cropped_frame = Mask_intesecrtion.region_of_interest(gray,1)
    if counter > 100:
        gray= corner_3(cropped_frame,1,counter)
   # ransac_method(x_points,y_points)
    
    cv2.imshow("result", gray)

    if cv2.waitKey(3) == ord('q'):
        break
    
cv2.destroyAllWindows()


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