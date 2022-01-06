import cv2
import numpy as np
import math
import time

import Lanes_mine

"""
This class implements the lane keeping algorithm by calculating angles from the detected lines slopes.
Input: lines from lane detection 
Output: angle
To do: get_poly_points inputs from lane detection // why sample_x/2 in get_error 
Ideas: to use y = ax^2 + bx + c in line detection to detect arcs 
"""

class LaneKeeping:

    lane_width_px = 478 

    def __init__(self):
        
        self.angle = 0.0
        self.last_time = 0
        self.last_error = 0
        self.right_fit = [0,0]
        self.left_fit = [0,0]
        self.last_PD = 0
        self.Kp = 0.1  # .06
        self.Kd = 0.02 # .01
        self.cache = [None, None]
        self.last_angle=0.0
    

    def canny(self, image):
        #Returns the processed image
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        blur=cv2.GaussianBlur(gray,(5,5),0)
        canny=cv2.Canny(blur,100,300)
        return canny

    def masked_region(self,image,height,width):
        #mask over the whole frame
        polygons=np.array([
            [(0,height),(width/4,height/4),(width*3/4,height/4),(width,height)] #(y,x)
            ])
        mask=np.zeros_like(image)
        cv2.fillPoly(mask,np.int32([polygons]),255)
        masked_image=cv2.bitwise_and(image,mask) 
        return masked_image

    def warp(self,image,height,width):
        #Transforms the image to bird-view(-ish)

        # Destination points for warping
        dst_points = np.float32([[0, height],[width, height],[0, 0],[width, 0]])
        src_points = np.float32([[0,height],[width,height],[width/4,height/4],[width*3/4,height/4] ])

        warp_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        inv_warp_matrix = cv2.getPerspectiveTransform(dst_points,src_points)

        warped_frame = cv2.warpPerspective(image,warp_matrix,(width,height))
        return warped_frame

    def make_coordinates(self,line_parameters,max_y):
        x1, x2, y1, y2 = 0, 0, 0, 0
        if len(line_parameters) != 0:
            slope,intercept=line_parameters
    
        y1=max_y
        y2=y1*(3/5)   
        x1=int((y1-intercept)/slope)
        x2=int((y2-intercept)/slope)
        return np.array([x1,y1,x2,y2])

    def make_average_lines(self,lines,width):
        left_fit=[]     
        right_fit=[]
        end_y=0
        right_line=np.zeros(4)
        left_line=np.zeros(4)
        boundary = 1.0/3.0
        #Left lane line segment should be on left 2/3 of the screen
        left_region_boundary = width * (1 - boundary)
        #Right lane line segment should be on right 2/3 of the screen
        right_region_boundary = width * boundary
        
        #If no line segments detected in frame, return an empty array
        if lines is None:
                return np.array([left_fit,right_fit])
        
        #Loop through the lines
        for line in lines:
            x1,y1,x2,y2=line[0]
            if (np.abs(x1-x2) < 0.06) | (np.abs(y2-y1) < 0.06):
                continue
            else:
                slope = (y2 - y1) / (x2 - x1)
                intercept =y1 - (slope * x1)
            if y1 > end_y:
                end_y=y1

            if (slope < 0) & (x1 < left_region_boundary):
                left_fit.append((slope,intercept))
            elif (slope > 0) & (x1 > right_region_boundary):
                right_fit.append((slope, intercept))
    
        left_fit_average=np.average(left_fit,axis=0)  
        right_fit_average=np.average(right_fit,axis=0)
        if len(left_fit) > 0:
            left_line=self.make_coordinates(left_fit_average,end_y)
        if len(right_fit) > 0:
            right_line=self.make_coordinates(right_fit_average,end_y)

        return left_fit_average, right_fit_average

    def display_lines(self,image,av_lines):
        line_image=np.zeros_like(image)
        if av_lines is not None:
            for line in av_lines:
                x1,y1,x2,y2 = line.reshape(4)
                cv2.line(line_image,(int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 10)
        return line_image

  
    def lanes_pipeline(self):

        cap=cv2.VideoCapture("straight_line & roundabout.mp4")
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        ret, fframe = cap.read()

        while(cap.isOpened()):
            ret,frame=cap.read()
            if ret==False:
                break
            self.height=frame.shape[0]
            self.width=frame.shape[1]

            canny_image=self.canny(frame)
            wrapped_image=self.warp(canny_image,self.height,self.width)
            masked_image=self.masked_region(wrapped_image,self.height,self.width)
            # masked_image = apply_mask(wrapped_image, stencil)
            lines=cv2.HoughLinesP(masked_image,2,np.pi/180,20,np.array([]),minLineLength=5,maxLineGap=5)
            
            self.left, self.right=self.make_average_lines(lines,self.width)
            Angle=self.lane_keeping_pipeline(self.left, self.right)
            # lined_image=self.display_lines(wrapped_image,averaged_lines)
            # combo_image=cv2.addWeighted(wrapped_image,0.8,lined_image,1,1)

            # cv2.imshow('tt',combo_image)
            # if cv2.waitKey(25) & 0xFF==ord('q'):
            #     break

        cap.release()
        return Angle


    def get_poly_points(self, left_fit, right_fit):		#left_fit=[a,b,c] from y=ax^2+bx+c
        
        # TODO: CHECK EDGE CASES

        ysize = self.height

        # Get the points for the entire height of the image
        plot_y = np.linspace(0, ysize - 1, ysize)
        plot_xleft = left_fit[0] * plot_y  + left_fit[1] 
        plot_xright = right_fit[0] * plot_y  + right_fit[1] 

        return plot_xleft.astype(np.int), plot_xright.astype(np.int)

    def get_error(self, left_x, right_x): #the x calculated from get_poly_points

        factor = int(round(0.5 * self.height)) #240*0.5= 120

        sample_right_x = right_x[factor:]  #all after factor(120) => right_x[121,122,..240]
        sample_left_x = left_x[factor:] 	#all after factor => left_x[121,122,...240]  => the bottom of the frame
        sample_x = np.array((sample_right_x + sample_left_x) / 2.0)

        if len(sample_x) != 0:
            weighted_mean = self.weighted_average(sample_x)

        error = weighted_mean - int(self.width / 2.0)
        setpoint = weighted_mean
        #print("Center: ", int(self.width / 2.0), " Mean: ", weighted_mean)
        
        return error, setpoint

    def weighted_average(self, num_list):

        mean = 0
        count = len(num_list)
        # CHECK WEIGHTS, SHOULD BE FLIPPED?
        weights = [*range(0, count)]
        mean = np.average(num_list, weights=weights)

        return mean

    def plot_points(self, left_x, left_y, right_x, right_y, frame):

        out = frame * 0
        for i in range(len(left_x)):
            cv2.circle(out, (left_x[i], left_y[i]), 10, color=(255, 255, 255), thickness=-1)
            cv2.circle(out, (right_x[i], right_y[i]), 10, color=(255, 255, 255), thickness=-1)

        return out


    def lane_keeping_pipeline(self, Left, Right):
        start = time.time()
        # poly_image = frame
          
            #==============================================#from lane detection
        left, right= Left, Right
        nose2wheel = self.height 
        left_check=np.sum(left)
        right_check= np.sum(right)
        if not np.isnan(left_check) and not np.isnan(right_check):   
            print("BOTH LANES")

            self.cache[0] = left
            self.cache[1] = right
            left_x, right_x = self.get_poly_points(left, right)

            error, setpoint = self.get_error(left_x, right_x)
            # print("GetError: ", error)

            self.angle = 90 - math.degrees(math.atan2(nose2wheel, error))

        elif np.isnan(right_check) and not np.isnan(left_check):
            print("LEFT LANE")
            self.cache[0] = left

            x1 = left[0] * self.height  + left[1] 
            x2 = left[1]
            dx = x2 - x1

            self.angle = 90 - math.degrees(math.atan2(nose2wheel, dx))

        elif np.isnan(left_check) and not np.isnan(right_check):
            print("RIGHT LANE")
            self.cache[1] = right
            
            x1 = right[0] * self.height  + right[1] 
            x2 = right[1]
            dx = x2 - x1

            self.angle = 90 - math.degrees(math.atan2(nose2wheel, dx))

        else:
            print("No lanes found")


        if self.angle > 0 and np.abs(self.last_angle- self.angle) < 15:
            self.angle = min(23, self.angle)
        elif self.angle < 0 and np.abs(self.last_angle- self.angle) < 15:
            self.angle = max(-23, self.angle)
        
        self.last_angle = self.angle
        print(self.angle, '\n')

        return self.angle

if __name__ == "__main__":
    lk= LaneKeeping()
    lk.lanes_pipeline()