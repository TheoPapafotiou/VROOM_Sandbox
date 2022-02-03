# edge detection 
from cgitb import grey
from cmath import sqrt
import numpy as np
from re import X
from xml.etree.ElementPath import xpath_tokenizer
import cv2
import matplotlib.pyplot as plt 
from matplotlib import pyplot as plt
import numpy as np
from numpy.core.overrides import verify_matching_signatures
from masksForIntersection import Mask_intesecrtion
from skimage.measure import ransac, LineModelND
from sklearn import linear_model
import time
import skimage
import os
import datetime
from timeit import timeit

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

"""
    THIS CLASS DETECT THE CORNER POINTS AND EGDES
    TRY TO DRAW THE LINE THAT THE CAR NEEDS TO FOLLOW IN ORDER TO NAVIGATE THOUGH THE INTERSECTION
    APPLIED RANSAC ALGORITHM TO FIND THE PATH LINE WITH THE AID OF THE DETECTED POINTS 
"""

class Intersection:

    def __init__(self):
        self.x_points = []
        self.y_points = []

## slow
    def ransac_method(self):
        data = np.column_stack([self.x_points,self.y_points])
        model = LineModelND()
        model.estimate(data)
        model_robust, inliers = ransac(data, LineModelND, min_samples=5,
                                residual_threshold=2, max_trials=1000)
        outliers = inliers == False 
        line_x = self.x_points
        line_y = model.predict_y(line_x)
        line_y_robust = model_robust.predict_y(line_x)
        
        ## COMMENTS FOR COMPREHENSION 
        # min_samples =  num of points selected in each iteratios
        # residual_threshold = if the ditsance of a point from a line is below a value then the point is classidied as an inlier otherwise outiler
        # max_trials = maximum number of iterations
        #LineModelND = Total least squares estimator for N-dimensional lines.
##In contrast to ordinary least squares line estimation, this estimator minimizes the orthogonal distances of points to the estimated lin

        ## REPRESENT THE RESULTS 
        # fig, ax = plt.subplots()
        # ax.plot(data[inliers, 0], data[inliers, 1], '.b', alpha=0.6,
        #         label='Inlier data')
        # ax.plot(data[outliers, 0], data[outliers, 1], '.r', alpha=0.6,
        #         label='Outlier data')
        # ax.plot(line_x, line_y, '-k', label='Line model from all data')
        # ax.plot(line_x, line_y_robust, '-b', label='Robust line model')
        # ax.legend(loc='lower left')
        # ax.set_title('MAXTRIALs = 2000')
        # plt.show()
        return line_x, line_y_robust


### slow

    def ransac4(self):

        X = np.array(self.x_points).reshape(-1, 1)
        y = np.array(self.y_points).reshape(-1, 1)
        # Create an instance of RANSACRegressor
        
        ransac = RANSACRegressor(base_estimator=LinearRegression(),
                                min_samples=50, max_trials=100,
                                loss='absolute_error', random_state=42,
                                residual_threshold=10)
        # LinearRegression = Ordinary least squares Linear Regression.
        # loss = find the absolute error per sample 
        # if the loss on a sample is greater than the residual_threshold, then this sample is classified as an outlier.
        lr = linear_model.LinearRegression()
        lr.fit(X,y)
        # Fit the model
        ransac.fit(X, y)
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)

        # Predict data of estimated models
        line_X = np.arange(X.min(), X.max())[:, np.newaxis]
        line_y = lr.predict(line_X)
        line_y_ransac = ransac.predict(line_X)

        ## REPRESENT THE RESULTS 
        # lw = 2
        # plt.scatter(
        #     X[inlier_mask], y[inlier_mask], color="yellowgreen", marker=".", label="Inliers"
        # )
        # plt.scatter(
        #     X[outlier_mask], y[outlier_mask], color="gold", marker=".", label="Outliers"
        # )
        # plt.plot(line_X, line_y, color="navy", linewidth=lw, label="Linear regressor")
        # plt.plot(
        #     line_X,
        #     line_y_ransac,
        #     color="cornflowerblue",
        #     linewidth=lw,
        #     label="RANSAC regressor",
        # )
        # plt.legend(loc="lower right")
        # plt.xlabel("Input")
        # plt.ylabel("Response")
        # plt.title('R4-ms=50')
        # plt.show()
        return line_X, line_y_ransac


    def corner_3(self,gray,turn): #turn = 1 : straight, turn = 2 : small right, turn = 3 : left big
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
                self.x_points.append(x)
                self.y_points.append(y)            
                cv2.circle(gray, (x, y), 3, 255, -1)
            elif turn == 2:
                if x == x2 and y == y2:
                    self.x_points.append(x)
                    self.y_points.append(y) 
                    cv2.circle(gray, (x, y), 3, 255, -1)
        # xnew,ynew = ransac_method(x_points,y_points,counter)

        white = (255,255,255)

        ## FIRST METHOD

        # if len(x_points) > 2:
        #     cv2.line(gray, (0, int(height)),(int(xnew[0]), int(ynew[0])), white, 9) 

        ## SECOND METHOD
        # if len(x_points) > 2:
        #     cv2.line(gray, (0, int(height)),(self.x_points[1], self.y_points[1]), white, 9)

        return gray



    def pipeline(self):
        cap = cv2.VideoCapture("videos/straight.mp4")
        counter = 0
        times1 = list()
        times2 = list()
        while (cap.isOpened()):
            counter = counter +  1
            _,frame = cap.read()
            gray = Mask_intesecrtion.canny(frame)
            width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
            cropped_frame = Mask_intesecrtion.region_of_interest(gray,5)  
            gray= self.corner_3(cropped_frame,1)
            
            if counter > 200:
                start = time.time()
                self.ransac4()
                end = time.time()
                times2.append(end-start)
            # if counter > 200:
            #     start = time.time()
            #     self.ransac_method()
            #     end = time.time()
            #     times1.append(end-start)


            cv2.imshow("result", gray)
            if counter > 300:
                break
            if cv2.waitKey(3) == ord('q'):
                break

        ## FOR TIME ANALYSIS OF ALGO
        # iter = list()
        # for i  in range(1,counter+1-200):
        #     iter.append(i)

        # plt.xlabel('List Length')
        # plt.ylabel('Time Complexity')
        # plt.plot(iter,times2, label = 'ransac4Counter=200-300')
        # plt.plot(iter,times1,label = 'ransac_method1')
        # plt.grid()
        # plt.legend()
        # plt.show()
            
        cv2.destroyAllWindows()

if __name__ == "__main__":
    inte = Intersection()
    inte.pipeline()

