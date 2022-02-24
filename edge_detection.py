# edge detection 
from cgitb import grey
from cmath import sqrt
from tracemalloc import start
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
import skimage
import os
import datetime
import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import r2_score, mean_squared_error


"""
    THIS CLASS DETECT THE CORNER POINTS AND EGDES
    TRY TO DRAW THE LINE THAT THE CAR NEEDS TO FOLLOW IN ORDER TO NAVIGATE THOUGH THE INTERSECTION
    APPLIED RANSAC ALGORITHM TO FIND THE PATH LINE WITH THE AID OF THE DETECTED POINTS 
"""
"""
        THERE ARE THREE METHOD TO DETECT THE CORNER : 1. HARRIS CORNER SIMPLE(cornerHarris2), 2. HARRIS CORNER ADVANCED (cornerHarris1)
        3. SHI-TOMASI CORNER DETECTOR - GOOD FEATURES TO TRACK (FastCornerDetector)
"""
class Intersection:

    def __init__(self):
        self.x_points = []
        self.y_points = []
        self.white = (255,255,255)
        self.counter2 = 0
        self.data = []
        self.mindist = [] # the distance between two points for 


## slow
    def ransac_method(self):
        self.data = np.column_stack([self.x_points,self.y_points])
        model = LineModelND()
        if len(self.data) > 2:
            model.estimate(self.data)
            model_robust, inliers = ransac(self.data, LineModelND, min_samples=5,
                                    residual_threshold=2, max_trials=1000)
            outliers = inliers == False 
            line_x = self.x_points
            line_y = model.predict_y(line_x)
            line_y_robust = list(model_robust.predict_y(line_x))
            return line_x, line_y_robust
        else:
            return self.x_points,self.y_points

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

        return line_X, line_y_ransac


    def cornerHarris1(self,gray,turn): #turn = 1 : straight, turn = 2 : small right, turn = 3 : left big
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
        width  = gray.shape[1]   # float `width`
        height = gray.shape[0] 
        print(width,height)
        for i in res:
            x,y,x2,y2 = i.ravel() 
            if y > int(height)/3:
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

        return gray 

    def FastCornerDetector(self,img):
        fast = cv2.FastFeatureDetector_create()
        kp = fast.detect(img,None)
        img2 = cv2.drawKeypoints(img, kp,None, color=(255,0,0))
        return img2
    
    def cornerHarris2(self,img):
        corners = cv2.goodFeaturesToTrack(img,25,0.01,10)
        if corners is not None:
            corners = np.int0(corners)
            for i in corners:
                x,y = i.ravel()
                cv2.circle(img,(x,y),3,255,-1)
                self.x_points.append(x)
                self.y_points.append(y)
        return img

        ## DRAWING THE ARC FOR THE TURN
    def convert_arc(self,pt1, pt2, sagitta):

            # extract point coordinates
        x1, y1 = pt1
        x2, y2 = pt2

        # find normal from midpoint, follow by length sagitta
        n = np.array([y2 - y1, x1 - x2])
        n_dist = np.sqrt(np.sum(n**2))

        if np.isclose(n_dist, 0):
            # catch error here, d(pt1, pt2) ~ 0
            print('Error: The distance between pt1 and pt2 is too small.')

        n = n/n_dist
        x3, y3 = (np.array(pt1) + np.array(pt2))/2 + sagitta * n

        # calculate the circle from three points
        # see https://math.stackexchange.com/a/1460096/246399
        A = np.array([
            [x1**2 + y1**2, x1, y1, 1],
            [x2**2 + y2**2, x2, y2, 1],
            [x3**2 + y3**2, x3, y3, 1]])
        M11 = np.linalg.det(A[:, (1, 2, 3)])
        M12 = np.linalg.det(A[:, (0, 2, 3)])
        M13 = np.linalg.det(A[:, (0, 1, 3)])
        M14 = np.linalg.det(A[:, (0, 1, 2)])

        if np.isclose(M11, 0):
            # catch error here, the points are collinear (sagitta ~ 0)
            print('Error: The third point is collinear.')

        cx = 0.5 * M12/M11
        cy = -0.5 * M13/M11
        radius = np.sqrt(cx**2 + cy**2 + M14/M11)

        # calculate angles of pt1 and pt2 from center of circle
        pt1_angle = 180*np.arctan2(y1 - cy, x1 - cx)/np.pi
        pt2_angle = 180*np.arctan2(y2 - cy, x2 - cx)/np.pi

        return (cx, cy), radius, pt1_angle, pt2_angle
   
    def draw_ellipse(self,
    img, center, axes, angle,
    startAngle, endAngle, color,
    thickness=10, lineType=cv2.LINE_AA, shift=10):
    # uses the shift to accurately get sub-pixel resolution for arc
        # taken from https://stackoverflow.com/a/44892317/5087436
        center = (
            int(round(center[0] * 2**shift)),
            int(round(center[1] * 2**shift))
        )
        axes = (
            int(round(axes[0] * 2**shift)),
            int(round(axes[1] * 2**shift))
        )
        color = (255,255,255)
        return cv2.ellipse(
            img, center, axes, angle,
            startAngle, endAngle, color,
            thickness, lineType, shift)

    def min_dist(self,img,x_point,y_point):
        width  = img.shape[0] 
        return ((((x_point - int(width) )**2) + ((y_point)**2) )**0.5)




    def pipeline(self): # genika tha exo kai ayta ta input self.speed,self.angle, distanceFromHorizontal , cap = input ths kameras, route = 1,2,3 for the three options
        
        cap = cv2.VideoCapture("videos/try2_straight.mp4")
        counter = 0
        times1 = list()
        times2 = list()
        start = time.time()
        while (cap.isOpened()):
            
            time.sleep(0.5)
            counter = counter +  1
            _,frame = cap.read()
            gray = Mask_intesecrtion.canny(frame)
            width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
            img = Mask_intesecrtion.region_of_interest(gray,5)  

            
            method = 2 # 1 for harrisadvanced, 2 for simple harris, 3 for fast detector        
            
            route = 1 # For choosing the route the car follows -> route == 1: straight, route == 2: small right
            duration = time.time() - start
            print(duration)
            startH = time.time()
            if route == 1: #straight
                if duration > 3:
                # HARRIS CORNER DETECTION 
                    if method == 1:
                        img= self.cornerHarris1(gray,2)
                        cv2.putText(img, 'ADVANCED HARRIS', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_4)
                # FAST CORNER DETECTOR
                    if method == 3:       
                        cv2.putText(img, 'FAST CORNER DETECTOR', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_4)
                        img = self.FastCornerDetector(img)
                # SIMPLE HARRIS
                    if method == 2:
                        img = self.cornerHarris2(img)
                        cv2.putText(img, 'SIMPLE HARRIS', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_4)
                   
                    if self.counter2 < 5:            
                        self.x_points,self.y_points= self.ransac_method()
                        self.counter2 += 1                            
                    if self.counter2 == 5:
                        cv2.line(img, (int(width*0.1), int(height)),(int(self.x_points[0]), int(self.y_points[0])), self.white, 9)
                        cv2.line(img, (int(width*0.9), int(height)),(int(width-self.x_points[0]), int(self.y_points[0])), self.white, 9)
                        
            durationH = time.time()-startH
          
          
            if route == 2:
                ## FOR THE TURN -> DRAW ARC
                distance = 0 # Distance from horizontal line, for now fixed 
                if distance == 0:
                    # start = time.time()
                    # dt = 0                
                    # while (dt < 3.0):
                    #     # fix gia tora 
                    #     speed = 15
                    #     v = speed / 0.66 # R of arc 
                    #     angle = v * dt
                    #     dt = start - time.time()
                   
                   # if counter > 100:
            #SIMPLE HARRIS
                   if duration > 1: # limit : 5 is for u =  10 
                        if method == 2:
                            img = self.cornerHarris2(img)
                            cv2.putText(img, 'SIMPLE HARRIS', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_4)
                        if method == 1:
                            img= self.cornerHarris1(gray,2)
                            cv2.putText(img, 'ADVANCED HARRIS', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_4)

                            if self.counter2 < 5:            
                                self.x_points,self.y_points= self.ransac_method()
                                self.counter2 += 1                            
                            if self.counter2 == 5:
                                min_distance= self.min_dist(img,0,0) # max initialization 
                                for i in range (1, self.counter2):
                                    if min_distance > self.min_dist(img,self.x_points[i],self.y_points[i]):
                                        xmin = self.x_points[i]
                                        ymin = self.y_points[i]
                                        min_distance = self.min_dist(img,xmin,ymin)
                            
                                pt1 = (int(width/2.9),int(height))
                                #pt2 = (self.x_points[int(counter/2)],self.y_points[int(counter/2)])
                                pt2 = (int(xmin),int(ymin))
                                sagitta = 17
                                center, radius, start_angle, end_angle = self.convert_arc(pt1, pt2, sagitta)
                                axes = (radius, radius)
                               # self.draw_ellipse(img, center, axes, 0, start_angle, end_angle, 255)

            cv2.imshow('r',img)
            print("HARRIS&RANSAC / frame = ", durationH)
 
            if cv2.waitKey(12) == ord('q'):
                break
            
        cv2.destroyAllWindows()

if __name__ == "__main__":
    inte = Intersection()
    start2 = time.time()
    inte.pipeline()
    print("in total = ", time.time()-start2)





# 61 : 
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

#ransac 4:

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

#pipeline

            ##RANSAC
            # if counter > 200:
            #     start = time.time()
            #     self.ransac4 ()            cv2.line(img, (0, int(height)),(self.x_points[1], self.y_points[1]), self.white, 9)

            #     end = time.time()
            #     times2.append(end-start)
            # if counter > 200:
            #     start = time.time()
            #     self.ransac_method()
            #     end = time.time()
            #     times1.append(end-start)

#pipeline(after break):

         #   return self.angle 

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
