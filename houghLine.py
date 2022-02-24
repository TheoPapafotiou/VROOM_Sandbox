from __future__ import print_function
from re import X
import cv2
import numpy as np
import cv2 as cv
import numpy as np
import time
# =============== METHOD : DETECT LINES WITH HOUGHLINESP ===================== 

class Corner_HoughLines:
    def __init__(self):
        self.right = []
        self.left = []

    def make_lines(self,lines,newwidth,start):
            xl = []
            yl = []
            xr = []
            yr = []
            leftfit = []
            rightfit= []
            if lines is None:
                return np.array(leftfit), np.array(rightfit)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                print('x1,x2',x1,x2)
                if (x1<(newwidth*2/3+start)) and (x2<(newwidth*4/5+start)):
                    # 
                    yl.append(y1)
                    yl.append(y2)
                    xl.append(x1)
                    xl.append(x2)
                    # pts = np.array([[x1, y1], [x2, y2]], np.int32)
                    # cv2.polylines(self.img, [pts], True, (255, 0, 0), 4)
                else:
                    yr.append(y1)
                    yr.append(y2)
                    xr.append(x1)
                    xr.append(x2)
                    # pts = np.array([[x1, y1], [x2, y2]], np.int32)
                    # cv2.polylines(self.img, [pts], True, (0, 0, 255), 4)

            # cv2.imshow('lines',self.img)
            SlopeL, MiddleL = np.polyfit(xl, yl, 1)
            SlopeR, MiddleR = np.polyfit(xr, yr, 1)
            rightfit.append((SlopeR,MiddleR))
            leftfit.append((SlopeL,MiddleL))
        
            return leftfit,rightfit     

    def masked(self,image,height,width):  
            polygons=np.array([
                [(int(width/3), int(height*0.55)),(width,int(height*0.55)),(width,height*0.8),(int(width/3),height*0.8)] #(y,x)
                ])
            mask = np.zeros_like(image)
            cv2.fillPoly(mask, np.int32([polygons]), 255)
            MaskedImage = cv2.bitwise_and(image, mask)
            return MaskedImage

    def canny(self, image):
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            canny = cv2.Canny(blur, 150, 350)
            return canny

    def intersection(self,left,right):
            a1,b1=left[0]
            a2,b2=right[0]

            A = np.array([[-a1,1],[-a2, 1]])
            b = np.array([[b1], [b2]])
            xi, yi = np.linalg.solve(A, b)
            xi, yi = int(np.round(xi)), int(np.round(yi))
            print('(xi,yi)',xi,yi)

            return xi, yi

    def thresh_callback(self,maskedimg):
        
        contours= cv.findContours(maskedimg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours=contours[0] if len(contours)==2 else contours[1]
        count=0
        for c in contours:
            count=count+1
            x,y,w,h=cv2.boundingRect(c)
            if h > 10: 
                cv2.rectangle(self.img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.imshow('bounding box',self.img)
            if count==1:
                xmax=x 
                wmax=w 
        print('start, newwidth',xmax,wmax)
        return xmax,wmax

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


    def pipeline(self):
        cap=cv2.VideoCapture("videos/try1_smallright.mp4")
        while(cap.isOpened()):
            startD = time.time()

            ret,frame=cap.read()
            if ret==False:
                break
            height=frame.shape[0]
            width=frame.shape[1]
            # img = cv2.imread("test3.png")
            # height=img.shape[0]
            # width=img.shape[1]
            self.img=frame
            cannyimg= self.canny(frame)
            maskedimg=self.masked(cannyimg,height,width)

            lines = cv2.HoughLinesP(maskedimg, rho=2, theta=np.pi/180, threshold=20, lines=np.array([]),
                                        minLineLength=3, maxLineGap=40)
            start,newwidth=self.thresh_callback(maskedimg)
            left,right = self.make_lines(lines,newwidth,start)
            a,b= self.intersection(left,right)

            cv2.circle(maskedimg, (a,b), 10, color=(255, 0, 0), thickness=-1)
            duration = time.time() - startD 
            # for the arc: 
            pt1 = (int(height*0.34),int(width))
            pt2 = (a,b)
            #sagitta = 50
            sagitta = 50


            center, radius, start_angle, end_angle = self.convert_arc(pt1, pt2, sagitta)
            print(end_angle)
            axes = (radius, radius)
            self.draw_ellipse(maskedimg, center, axes, 0, start_angle, end_angle, 255)
            print("DURATION / FRAME", duration)
            time.sleep(0.5)
            cv2.imshow('detected corner',maskedimg)
            if cv2.waitKey(25) & 0xFF==ord('q'):
                break
        cap.release()

if __name__ == "__main__":
    lk= Corner_HoughLines()
    lk.pipeline()
