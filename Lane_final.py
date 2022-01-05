import cv2
import numpy as np
import math
from numpy.core.fromnumeric import size
from numpy.lib.function_base import average

from Mask import Mask


class Lane_detection:
    right_line_points = []
    left_line_points = []
    right_line_slope = None
    left_line_slope = None
    right_slope_buffer = []
    left_slope_buffer = []
    right_moving_slope = None
    left_moving_slope = None

    right_points_buffer = []
    left_points_buffer = []

    lines = np.empty((1, 1, 1))

    def __init__(self, vp, use_mask_class=False, mask_filename=None):
        self.cap = cv2.VideoCapture(vp)
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        ret, fframe = self.cap.read()
        if use_mask_class:
            m = Mask(fframe, mask_filename)
            self.stencil = m.stencil
        else:
            self.stencil = self.make_stencil(fframe, fframe.shape[0], fframe.shape[1])

    def canny(self, image):
        # Returns the processed image
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, 100, 300)
        return canny

    def make_stencil(self, image, height, width):
        # mask over the whole frame
        polygons = np.array([
            [(0, height), (width / 4, height / 4), (width * 3 / 4, height / 4), (width, height)]  # (y,x)
        ])
        canny_image = self.canny(image)
        stencill = np.zeros_like(canny_image)
        cv2.fillPoly(stencill, np.int32([polygons]), 255)
        return stencill

    def detect(self, output_filename=None):
        if output_filename is not None:
            feed_obj_out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*"XVID"), 30.0,
                                           (self.frame_width, self.frame_height), True)
        while True:
            ret, frame = self.cap.read()
            if ret == False:
                break
            height = frame.shape[0]
            width = frame.shape[1]

            canny_image = self.canny(frame)
            wrapped_image = self.warp(canny_image, height, width)
            masked_image = self.apply_mask(wrapped_image, self.stencil)
            self.lines = cv2.HoughLinesP(masked_image, 2, np.pi / 180, 20, np.array([]), minLineLength=5, maxLineGap=5)

            averaged_lines, info = self.make_average_lines(self.lines, width, wrapped_image)

            # all_lines_image = self.all_lines_found(lines, width, wrapped_image)

            # colored_wrapped_img = cv2.cvtColor(wrapped_image, cv2.COLOR_GRAY2BGR)
            # colored_all_lines_img = self.all_lines_found(lines, width, colored_wrapped_img)
            # test, testinfo = make_average_lines_2(lines, width, wrapped_image)
            # lined_image = self.display_lines(wrapped_image, averaged_lines)
            lined_image = self.display_lines(wrapped_image, [self.right_line_points, self.left_line_points])
            combo_image = cv2.addWeighted(wrapped_image, 0.8, lined_image, 1, 1)

            # combo_img_all_lines = cv2.addWeighted(colored_wrapped_img
            #                                       , 0.8, colored_all_lines_img, 0.8, 0.5)
            print(info)
            cv2.imshow('tt', combo_image)
            # cv2.waitKey()
            # if output_filename is not None:
            #     feed_obj_out.write(combo_img_all_lines)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        self.cap.release()
        if output_filename is not None:
            feed_obj_out.release()
        cv2.destroyAllWindows()
        # out.release()

    def apply_mask(self, image, stencill):
        masked_imagee = cv2.bitwise_and(image, stencill)
        return masked_imagee

    def warp(self, image, height, width):
        # Transforms the image to bird-view(-ish)

        # Destination points for warping
        dst_points = np.float32([[0, height], [width, height], [0, 0], [width, 0]])
        src_points = np.float32([[0, height], [width, height], [width / 4, height / 4], [width * 3 / 4, height / 4]])

        warp_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        inv_warp_matrix = cv2.getPerspectiveTransform(dst_points, src_points)

        warped_frame = cv2.warpPerspective(image, warp_matrix, (width, height))
        return warped_frame

    def warp_nassos(self, image, intensity):  # DOESNT WORK ALLA GENIKA XREIAZETAI MIA WARP ALLA ME INTENSITY
        rect = [[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]]
        rect = np.array(rect)
        dst = np.array([
            [0, 0],
            [self.width - intensity * self.width / 10, 0],
            [self.width - intensity * self.width / 10, self.height - intensity * self.height / 10],
            [0, self.height - intensity * self.height / 10]], dtype="float32")
        t = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, t, (self.width, self.height))
        return warped

    def make_coordinates(self, line_parameters, max_y):
        x1, x2, y1, y2 = 0, 0, 0, 0
        if len(line_parameters) != 0:
            slope, intercept = line_parameters

        y1 = max_y
        y2 = y1 * (3 / 5)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])

    def make_coordinates2(self, line_parameters, max_y):
        x1, x2, y1, y2 = 0, 0, 0, 0
        if len(line_parameters) != 0:
            slope, intercept = line_parameters

        y1 = max_y
        y2 = 0
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([int(x1), int(y1), int(x2), int(y2)])

    def filter_lines(self, slope_threshold=0.2, image_if_debug=None):
        info = {}
        info["found_lines"] = len(self.lines)
        if self.lines is None:
            return -1
        kept_distances = []
        lengths = self.calculate_line_lenght(self.lines)
        avg_length = np.average(lengths)
        info["avg_length"] = avg_length

        slopes = self.calculate_line_slope(self.lines)
        # for i in range(len(lengths)):
        #     print(slopes[i])
        #     self.display_lines_on_img(image_if_debug, lines=self.lines[i], wait=True)

        ## direction array is an array of the same length as the lines.
        ##  There is an 1 when the line is left, a 2 when the line is right
        #       and 0 when the line is **perfectly** horizontal
        direction_array = []

        for i in range(len(slopes)):
            if slopes[i] < 0 and self.lines[i][0][1] < self.width / 2:
                direction_array.append(1)
            elif slopes[i] > 0 and self.lines[i][0][1] > self.width / 2:
                direction_array.append(2)
            else:
                direction_array.append(0)

        right_lane_slopes = []
        left_lane_slopes = []
        for s in slopes:
            if s < -0.2:
                left_lane_slopes.append(s)
            elif s > 0.2:
                right_lane_slopes.append(s)

        avg_right_slope = np.average(np.array(right_lane_slopes))
        avg_left_slope = np.average(np.array(left_lane_slopes))

        self.moving_average_slopee(avg_right_slope, avg_left_slope)

    def moving_average_slopee(self, right_slope, left_slope, moving_size=10):
        weights = [*range(1, moving_size + 1)]
        r_weights = weights
        l_weights = weights

        if len(self.right_slope_buffer) >= moving_size:
            self.right_slope_buffer.pop(0)
            self.right_slope_buffer.append(right_slope)
        else:
            self.right_slope_buffer.append(right_slope)
            r_weights = [*range(1, len(self.right_slope_buffer) + 1)]

        self.right_moving_slope = np.average(self.right_slope_buffer, weights=r_weights)
        if len(self.left_slope_buffer) >= moving_size:
            self.left_slope_buffer.pop(0)
            self.left_slope_buffer.append(left_slope)
        else:
            self.left_slope_buffer.append(left_slope)
            l_weights = [*range(1, len(self.left_slope_buffer) + 1)]
        self.left_moving_slope = np.average(self.left_slope_buffer, weights=l_weights)

        return self.right_moving_slope, self.left_moving_slope

    def show_avg_slope_in_static_lines_for_pi_cam(self, img, wait):
        lx1 = 11
        ly1 = 1092
        ly2 = 350
        lx2 = (ly2 - ly1 - self.left_moving_slope * lx1) / self.left_moving_slope
        left_line = [lx1, ly1, lx2, ly2]

        self.display_lines_on_img(img, [left_line], wait=wait)

    def all_lines_found(self, lines, width, image):
        left_fit = []
        right_fit = []
        end_y = 0
        right_line = np.zeros(4)
        left_line = np.zeros(4)
        # horizontal_lines=[]
        boundary = 1.0 / 3.0
        # Left lane line segment should be on left 2/3 of the screen
        left_region_boundary = width * (1 - boundary)
        # Right lane line segment should be on right 2/3 of the screen
        right_region_boundary = width * boundary

        # If no line segments detected in frame, return an empty array
        if lines is None:
            return np.array([left_fit, right_fit])

        # Loop through the lines
        count = 0
        lined_image = image.copy()

        for line in lines:
            x1, y1, x2, y2 = line[0]

            if (np.abs(x1 - x2) < 0.06) | (np.abs(y2 - y1) < 0.06):
                continue
            else:
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - (slope * x1)
            if y1 > end_y:
                end_y = y1

            if (slope < 0) & (x1 < left_region_boundary):
                left_fit.append((slope, intercept))
            elif (slope > 0) & (x1 > right_region_boundary):
                right_fit.append((slope, intercept))
        # if len(horizontal_lines > 0):
        #     get_horizontal_lanes(horizontal_lines)

        right_lines = []
        left_lines = []
        for rl in right_fit:
            right_lines.append(self.make_coordinates2(rl, end_y))
            # left_line = self.make_coordinates(left_fit_average, end_y)
        for ll in left_fit:
            left_lines.append(self.make_coordinates2(ll, end_y))
            # left_line = self.make_coordinates(left_fit_average, end_y)

        lined_image = self.display_lines_left_right(image, left_lines, right_lines, 2)
        return lined_image

    # def get_horizontal_lanes(lines):
    #     for line in lines:
    #         x1,y1,x2,y2=line[0]
    #     return horizontal_lines
    def make_average_lines_2(self, lines, width, image):
        info = {}
        left_fit = []
        right_fit = []
        end_y = 0
        right_line = np.zeros(4)
        left_line = np.zeros(4)
        # horizontal_lines=[]
        boundary = 1.0 / 3.0
        # Left lane line segment should be on left 2/3 of the screen
        left_region_boundary = width * (1 - boundary)
        # Right lane line segment should be on right 2/3 of the screen
        right_region_boundary = width * boundary

        # If no line segments detected in frame, return an empty array
        if lines is None:
            return np.array([left_fit, right_fit])

        # Loop through the lines
        info["num_of_lines"] = len(lines)
        apoklisi = 0
        kept_apoklisi = 0
        count = 0
        lined_image = image.copy()
        for l in lines:
            self.display_lines(lined_image, l)

        cv2.imshow("ttttt", lined_image)
        cv2.waitKey()

        for line in lines:
            x1, y1, x2, y2 = line[0]
            apoklisi += np.abs(x1 - x2) + np.abs(y2 - y1)
            kept_apoklisi = apoklisi
            if (np.abs(x1 - x2) < 0.06) | (np.abs(y2 - y1) < 0.06):
                # horizontal_lines=line[0]
                kept_apoklisi -= np.abs(x1 - x2) + np.abs(y2 - y1)
                count += 2
                continue
            else:
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - (slope * x1)
            if y1 > end_y:
                end_y = y1

            if (slope < 0) & (x1 < left_region_boundary):
                left_fit.append((slope, intercept))
            elif (slope > 0) & (x1 > right_region_boundary):
                right_fit.append((slope, intercept))
        info["avg_total_apoklisi"] = round(apoklisi / (2 * len(lines)), 3)
        info["avg_kept_apoklisi"] = round(kept_apoklisi / (2 * len(lines) - count), 3)
        info["not_kept"] = round(count / 2, 3)
        # if len(horizontal_lines > 0):
        #     get_horizontal_lanes(horizontal_lines)

        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        if len(left_fit) > 0:
            left_line = self.make_coordinates(left_fit_average, end_y)
        if len(right_fit) > 0:
            right_line = self.make_coordinates(right_fit_average, end_y)

        return (left_line, right_line), info

    def make_average_lines(self, lines, width, image, moving_avg_frames=10):
        info = {}
        left_fit = []
        right_fit = []
        end_y = 0
        right_line = np.zeros(4)
        left_line = np.zeros(4)
        # horizontal_lines=[]
        boundary = 1.0 / 3.0
        # Left lane line segment should be on left 2/3 of the screen
        left_region_boundary = width * (1 - boundary)
        # Right lane line segment should be on right 2/3 of the screen
        right_region_boundary = width * boundary

        # If no line segments detected in frame, return an empty array
        if lines is None:
            return np.array([left_fit, right_fit])

        # Loop through the lines
        info["num_of_lines"] = len(lines)
        apoklisi = 0
        kept_apoklisi = 0
        count = 0

        for line in lines:
            x1, y1, x2, y2 = line[0]
            apoklisi += np.abs(x1 - x2) + np.abs(y2 - y1)
            kept_apoklisi = apoklisi

            ## einai ints, to 0.06 den bgazei nohma. apla kanei exclude ta idia
            if (np.abs(x1 - x2) < 0.06) | (np.abs(y2 - y1) < 0.06):
                # horizontal_lines=line[0]
                kept_apoklisi -= np.abs(x1 - x2) + np.abs(y2 - y1)
                count += 2
                continue
            else:
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - (slope * x1)
            if y1 > end_y:
                end_y = y1

            if (slope < 0) & (x1 < left_region_boundary):
                left_fit.append((slope, intercept))
            elif (slope > 0) & (x1 > right_region_boundary):
                right_fit.append((slope, intercept))
        info["avg_total_apoklisi"] = round(apoklisi / (2 * len(lines)), 3)
        info["avg_kept_apoklisi"] = round(kept_apoklisi / (2 * len(lines) - count), 3)
        info["not_kept"] = round(count / 2, 3)
        # if len(horizontal_lines > 0):
        #     get_horizontal_lanes(horizontal_lines)

        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)

        info["left_lines"] = len(left_fit)
        info["right_lines"] = len(right_fit)
        ########################
        ### SHOW FRAME BY FRAME
        # line_image = np.zeros_like(image)
        # points = self.make_coordinates2(left_fit_average, end_y)
        # cv2.line(line_image, (points[0], points[1]), (points[2], points[3]), (255, 0, 0), 3)
        # combo_img_all_lines = cv2.addWeighted(image
        #                                       , 0.8, line_image, 0.8, 0.5)
        # cv2.imshow("1 line", combo_img_all_lines)
        # cv2.waitKey()
        ###
        #########################

        if len(self.right_slope_buffer) < moving_avg_frames:
            self.right_slope_buffer.append(right_fit_average[0])
            self.right_moving_slope = right_fit_average[0]
        else:
            self.right_slope_buffer.pop(0)
            self.right_slope_buffer.append(right_fit_average[0])
            self.right_moving_slope = (self.right_moving_slope + sum(self.right_slope_buffer)) / \
                                      (moving_avg_frames + 1)

        if len(self.left_slope_buffer) < moving_avg_frames:
            self.left_slope_buffer.append(left_fit_average[0])
            self.left_moving_slope = left_fit_average[0]
        else:
            self.left_slope_buffer.pop(0)
            self.left_slope_buffer.append(left_fit_average[0])
            self.left_moving_slope = (self.left_moving_slope + sum(self.left_slope_buffer)) / \
                                     (moving_avg_frames + 1)

        if len(left_fit) > 0:
            self.left_line_points = self.make_coordinates2((self.left_moving_slope, left_fit_average[1]), end_y)
        if len(right_fit) > 0:
            self.right_line_points = self.make_coordinates2((self.right_moving_slope, right_fit_average[1]), end_y)
        # if len(left_fit) > 0:
        #     self.left_line_points = self.make_coordinates(left_fit_average, end_y)
        #     self.left_line_slope = left_fit_average[0]
        # if len(right_fit) > 0:
        #     right_line = self.make_coordinates(right_fit_average, end_y)
        #     self.left_line_slope = right_fit_average[0]

        return (left_line, right_line), info

    def add_lines_on_buf(self, right_line_points, left_line_points, buff_size):
        if len(self.right_points_buffer) == len(buff_size):
            self.right_points_buffer.pop(0)
            self.right_points_buffer.append()

    def moving_average_slope(self, r_average_fit, l_average_fit, moving_average_quantity):
        if len(self.right_slope_buffer) < moving_average_quantity:
            self.right_slope_buffer.append(r_average_fit[0])
            self.right_moving_slope = r_average_fit[0]
        else:
            self.right_slope_buffer.pop(0)
            self.right_slope_buffer.append(r_average_fit[0])
            self.right_moving_slope = (self.right_moving_slope + sum(self.right_slope_buffer)) / \
                                      (moving_average_quantity + 1)

        if len(self.left_slope_buffer) < moving_average_quantity:
            self.left_slope_buffer.append(l_average_fit[0])
            self.left_moving_slope = l_average_fit[0]
        else:
            self.left_slope_buffer.pop(0)
            self.left_slope_buffer.append(l_average_fit[0])
            self.left_moving_slope = (self.left_moving_slope + sum(self.left_slope_buffer)) / \
                                     (moving_average_quantity + 1)

    # def moving_average_points(self, ):

    def display_lines_left_right(self, image, left, right, thickness=10):
        line_image = np.zeros_like(image)
        if left is not None:
            for line in left:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (71, 53, 232), thickness=thickness)
        if right is not None:
            for line in right:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (235, 64, 52), thickness=thickness)

        return line_image

    def display_lines(self, image, av_lines, thickness=10):
        line_image = np.zeros_like(image)
        if av_lines is not None:
            for line in av_lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), thickness=thickness)
        return line_image

    def display_lines_on_img(self, img, lines, thickness=10, wait=True):
        line_image = np.zeros_like(img)
        # if lines[0].size == 0:
        #     cv2.imshow("lines", img)
        #     if wait:
        #         cv2.waitKey()
        # else:
        for line in lines:
            # print(line)
            pass
        try:
            if len(lines) != 1:
                for line in lines:
                    # x1, y1, x2, y2 = line.reshape(4)
                    x1, y1, x2, y2 = line[0]
                    cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), thickness=thickness)
                    combined_image = cv2.addWeighted(img, 0.8, line_image, 1, 1)
                    cv2.imshow("lines", combined_image)
                    if wait:
                        cv2.waitKey()
            else:
                x1, y1, x2, y2 = lines[0]
                cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), thickness=thickness)
                combined_image = cv2.addWeighted(img, 0.8, line_image, 1, 1)
                cv2.imshow("lines", combined_image)
                if wait:
                    cv2.waitKey()

            # if line.size == 0:
            #     raise
        except:
            cv2.imshow("lines", img)
            if wait:
                cv2.waitKey()

    def calculate_line_lenght(self, lines):
        lengths = []
        if lines is None:
            return np.array(lengths)
        if len(lines) == 1:
            x1, y1, x2, y2 = lines[0]
            l = math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))
            lengths.append(l)
            return np.array(lengths)
        else:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                l = math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))
                lengths.append(l)
            return np.array(lengths)

    def calculate_line_slope(self, lines):
        slopes = []
        if lines is None:
            return np.array(slopes)
        if len(lines) == 1:
            try:
                x1, y1, x2, y2 = lines[0]
                s = (y2 - y1) / (x2 - x1)
                slopes.append(s)
                return np.array(slopes)
            except:
                x1, y1, x2, y2 = lines[0][0]
                s = (y2 - y1) / (x2 - x1)
                slopes.append(s)
                return np.array(slopes)
        else:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                s = (y2 - y1) / (x2 - x1)
                slopes.append(s)
            return np.array(slopes)

    def is_turning(self):
        slopes = self.calculate_line_slope(self.lines)
        direction_array = []

        for i in range(len(slopes)):
            if self.lines[i][0][1] < self.width / 2:
                direction_array.append(1)  # left lines
            elif self.lines[i][0][1] > self.width / 2:
                direction_array.append(2)
            else:
                direction_array.append(0)
        right_slopes = []
        left_slopes = []
        for i in range(len(direction_array)):
            # print(f"slope: {slopes[i]}")
            # self.display_lines_on_img(img, lines=self.lines[i])

            if direction_array[i] == 1:
                if self.lines[i][0][1] < self.width / 2:
                    left_slopes.append(slopes[i])
            elif direction_array[i] == 2:
                if self.lines[i][0][1] > self.width / 2:
                    right_slopes.append(slopes[i])
        all_votes = len(right_slopes) + len(left_slopes) + 1
        right_votes = 0
        left_votes = 0
        for ls in left_slopes:
            if ls > 0:
                left_votes += 1
            if -0.1 > ls > -0.5:
                right_votes += 1
        for rs in right_slopes:
            if rs < 0:
                right_votes += 1
            if 0.1 < rs < 0.5:
                left_votes += 1

        right_confidence = right_votes / all_votes
        left_confidence = left_votes / all_votes

        if right_confidence > left_confidence:
            if right_confidence > 0.8:
                return 1
            else:
                return -1
        else:
            if left_confidence > 0.8:
                return 2
            else:
                return -1

    ####################################################
    # ------------HORIZONTAL-LINE-DETECTION-------------#
    ####################################################

    def detect2(self, output_filename=None):
        min_line_length = 20
        while True:
            ret, frame = self.cap.read()

            self.height = frame.shape[0]
            self.width = frame.shape[1]

            canny_image = self.canny(frame)
            warped_image = self.warp(canny_image, self.height, self.width)
            # warped_image = self.warp_nassos(canny_image, intensity=1)
            masked_image = self.apply_mask(warped_image, self.stencil)

            # M = Mask(warped_image, "raspi_test.json")

            self.lines = cv2.HoughLinesP(masked_image, 2, np.pi / 180, 20, np.array([]), minLineLength=min_line_length)
            # self.is_turning(masked_image)
            # self.display_lines_on_img(masked_image, lines=self.lines)
            # self.filter_lines(warped_image)

            # self.show_avg_slope_in_static_lines_for_pi_cam(warped_image, wait=False)

            # for i in range(1, 10):
            #     for j in range(1,36):
            #         self.lines = cv2.HoughLinesP(masked_image, i, np.pi / (10*j), 20, np.array([]), minLineLength=5, maxLineGap=5)
            #         print(f"rho  : {i}, \ntheta: {np.pi / (10*j)}\nlines: {len(self.lines)}\n\n")
            is_turning = self.is_turning()
            horizontal_lines, detected_bool, detect_intensity = self.detect_horizontal(masked_image)

            if is_turning == 1:
                print("right turn")
            elif is_turning == 2:
                print("left turn")
            if detected_bool:
                print(f"DETECTED, intensity: {detect_intensity}")
            self.display_lines_on_img(warped_image, horizontal_lines, thickness=10, wait=False)
            # cv2.imshow('tt', combo_image)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    def detect_horizontal(self, image, slope_threshold=0.02, screen_fractions=5):
        detected = []
        det_bool = False
        detection_dist_intensity = -1  # 1 to 3 metric of how close is the detected line
        if self.lines is None:
            return np.array(detected), det_bool, detection_dist_intensity
        if self.is_turning() < 0:
            for line in self.lines:
                # print(line)
                # self.calculate_line_slope(line)
                # print(self.calculate_line_slope(line))
                # print(self.calculate_line_lenght(line)[0])
                x1, y1, x2, y2 = line[0]

                # self.display_lines_on_img(image, line, thickness=10)
                if self.calculate_line_lenght(line)[0] > self.width / 5:  # megalytero apo to 1/5 ths eikonas
                    if abs(self.calculate_line_slope(line)[0]) < slope_threshold:
                        detected.append(line)
        avg_y = -1
        if len(detected) > 0:
            sum = 0
            det_bool = True
            for d in detected:
                sum += d[0][1]
            avg_y = sum / len(detected)
            fraction = avg_y / self.height
            detection_dist_intensity = round(fraction * screen_fractions + 1, None)
            # if avg >= self.height / 3:
            #     detection_dist_intensity = 1
            # elif self.height / 3 > avg >= 2 * self.height / 3:
            #     detection_dist_intensity = 2
            # else:
            #     detection_dist_intensity = 3

        return np.array(detected), det_bool, detection_dist_intensity, avg_y


# lk = Lane_detection("real_tests_picam/straight_line.mp4", use_mask_class=True, mask_filename="raspi_test.json")
# lk = Lane_detection("real_tests_picam/straight _ turn.mp4", use_mask_class=True, mask_filename="raspi_test.json")
# lk = Lane_detection("real_tests_picam/random.mp4", use_mask_class=True, mask_filename="raspi_test.json")
lk = Lane_detection("real_tests_picam/straight_line _ roundabout.mp4", use_mask_class=True,
                    mask_filename="raspi_test.json")
lk.detect2()
