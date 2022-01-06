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
        self.lane_detection = Lanes_mine.Lanes_mine()

        # Rolling average parameters
        self.median_matrix = list()
        self.median_constant = 0
        self.median = 0.0       

    def get_poly_points(self, left_fit, right_fit):		#left_fit=[a,b,c] from y=ax^2+bx+c
        
        # TODO: CHECK EDGE CASES

        ysize = self.height

        # Get the points for the entire height of the image
        plot_y = np.linspace(0, ysize - 1, ysize)
        plot_xleft = left_fit[0] * plot_y  + left_fit[1] 
        plot_xright = right_fit[0] * plot_y  + right_fit[1] 

        # But keep only those points that lie within the image
        # plot_xleft = plot_xleft[(plot_xleft >= 0) & (plot_xleft <= xsize - 1)]
        # plot_xright = plot_xright[(plot_xright >= 0) & (plot_xright <= xsize - 1)]
        # plot_yleft = np.linspace(ysize - len(plot_xleft), ysize - 1, len(plot_xleft))
        # plot_yright = np.linspace(ysize - len(plot_xright), ysize - 1, len(plot_xright))

        return plot_xleft.astype(np.int), plot_xright.astype(np.int)

    def get_error(self, left_x, right_x): #the x calculated from get_poly_points

        # num_lines = 20
        # line_height = int(self.height / float(num_lines))
        # lines = np.flip(np.array([int(self.height - 1 - i * line_height) for i in range(num_lines)]))
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


    def lane_keeping_pipeline(self):
        start = time.time()
        # poly_image = frame
          
            #==============================================#from lane detection
        left, right,self.height, self.width=  self.lane_detection.lanes_pipeline() 
        # print("Polyfit: ", time.time() - start)
        nose2wheel = self.height 
        if left is not None and right is not None:   
            print("BOTH LANES")

            self.cache[0] = left
            self.cache[1] = right
            left_x, right_x = self.get_poly_points(left, right)
            #             print("Poly points: ", time.time() - start)

            #poly_image = self.plot_points(left_x, left_y, right_x, right_y, frame)
            #             print("Plot points: ", time.time() - start)

            error, setpoint = self.get_error(left_x, right_x)
            # print("GetError: ", error)

            self.angle = 90 - math.degrees(math.atan2(nose2wheel, error))

        elif right is None and left is not None:
            print("LEFT LANE")
            self.cache[0] = left

            x1 = left[0] * self.height ** 2 + left[1] * self.height + left[2]
            x2 = left[2]
            dx = x2 - x1

            self.angle = 90 - math.degrees(math.atan2(nose2wheel, dx))

        elif left is None and right is not None:
            print("RIGHT LANE")
            self.cache[1] = right
            
            x1 = right[0] * self.height ** 2 + right[1] * self.height + right[2]
            x2 = right[2]
            dx = x2 - x1

            self.angle = 90 - math.degrees(math.atan2(nose2wheel, dx))

        else:
            print("No lanes found")

        # # ===========TEST CODE=========
        # error = self.angle  # let's take angle=10
        # now = time.time()
        # dt = now - self.last_time

        # derivative = self.Kd * (error - self.last_error) / dt   #0.02*(10- 7)= 0.06
        # proportional = self.Kp * error   #0.1*10=1
        # PD = int(self.angle + derivative + proportional) #10+0.06+1 = 11.6

        # self.last_error = error
        # self.last_time = time.time()
        # self.median_matrix.insert(0, PD)

        # if len(self.median_matrix) == self.median_constant:         #why??? αφου εχει μονο το PD μεσα?
        #     self.median = np.average(self.median_matrix)   
        #     PD = self.median
        #     self.median = 0.0
        #     self.median_matrix.pop() #removes the last item

        # # PD /= 2
        # self.angle = PD
        # self.last_PD = PD

        # # font
        # font = cv2.FONT_HERSHEY_SIMPLEX

        # # org
        # org = (50, 50)

        # # fontScale
        # fontScale = 1

        # # Blue color in BGR
        # color = (255, 0, 0)

        # # Line thickness of 2 px
        # thickness = 1

        # # Using cv2.putText() method
        # image = cv2.putText(frame, 'Angle: ' + str(int(self.angle)), (50, 70), font,
        #                     fontScale, color, thickness, cv2.LINE_AA)

        # image = cv2.putText(frame, 'Derivative: ' + str(int(derivative)), (50, 90), font, fontScale, color, thickness,
        #                     cv2.LINE_AA)
        # cv2.putText(frame, 'Proportional: ' + str(int(proportional)), (50, 120), font, fontScale, color, thickness,
        #             cv2.LINE_AA)
        # cv2.putText(frame, 'CurrentVSfinal: ' + str(int(PD)) + ' ' + str(int(self.angle)), (0, 200), font, fontScale,
        #             color, thickness, cv2.LINE_AA)
        # # =========END TEST CODE======

        if self.angle > 0 and np.abs(self.last_angle- self.angle) < 15:
            self.angle = min(23, self.angle)
        elif self.angle < 0 and np.abs(self.last_angle- self.angle) < 15:
            self.angle = max(-23, self.angle)
        
        self.last_angle = self.angle
        print(self.angle, '\n')

        # return self.angle

if __name__ == "__main__":
    lk= LaneKeeping()
    cap=cv2.VideoCapture("straight_line.mp4")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    ret, fframe = cap.read()
    while(cap.isOpened()):
        lk.lane_keeping_pipeline()