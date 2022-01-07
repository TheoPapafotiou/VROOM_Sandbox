import cv2
import numpy as np
import math
from numpy.core.fromnumeric import size
from numpy.lib.function_base import average

from Mask import Mask


def canny(image):
    # Returns the processed image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 100, 300)
    return canny


def display_lines_on_img(img, lines, thickness=10, wait=True):
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


def calculate_line_lenght(lines):
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


def calculate_line_slope(lines):
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


def warp(image, height, width):
    # Transforms the image to bird-view(-ish)

    # Destination points for warping
    dst_points = np.float32([[0, height], [width, height], [0, 0], [width, 0]])
    src_points = np.float32([[0, height], [width, height], [width / 4, height / 4], [width * 3 / 4, height / 4]])

    warp_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    inv_warp_matrix = cv2.getPerspectiveTransform(dst_points, src_points)

    warped_frame = cv2.warpPerspective(image, warp_matrix, (width, height))
    return warped_frame


def apply_mask(image, stencill):
    masked_imagee = cv2.bitwise_and(image, stencill)
    return masked_imagee


class Detection:
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

    def __init__(self, vp, mask_filename=None):
        self.cap = cv2.VideoCapture(vp)
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        ret, fframe = self.cap.read()
        self.height = fframe.shape[0]
        self.width = fframe.shape[1]
        m = Mask(mask_filename)
        self.stencil = m.stencil

    def filter_lines(self, slope_threshold=0.2, image_if_debug=None):
        info = {}
        # info["found_lines"] = len(self.lines)
        if self.lines is None:
            return -1
        kept_distances = []
        # lengths = calculate_line_lenght(self.lines)
        # avg_length = np.average(lengths)
        # info["avg_length"] = avg_length

        slopes = calculate_line_slope(self.lines)
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
        for i in range(len(slopes)):
            display_lines_on_img(image_if_debug, lines=self.lines[i][0], wait=True)
            if slopes[i] < 0 and self.lines[i][0][1] < self.width / 2:
                left_lane_slopes.append(slopes[i])
            elif slopes[i] > 0 and self.lines[i][0][1] > self.width / 2:
                right_lane_slopes.append(slopes[i])

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


    def detect(self):
        min_line_length = 40
        while True:
            ret, frame = self.cap.read()
            canny_image = canny(frame)
            warped_image = warp(canny_image, self.height, self.width)
            masked_image = apply_mask(warped_image, self.stencil)
            self.lines = cv2.HoughLinesP(masked_image, 2, np.pi / 180, 20, np.array([]), minLineLength=min_line_length)
            self.filter_lines(image_if_debug=warped_image)
            RIGHT_SLOPE = self.right_moving_slope
            LEFT_SLOPE = self.left_moving_slope

            print(f"right slope: {RIGHT_SLOPE}")
            print(f"left  slope: {LEFT_SLOPE}")

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    ####################################################
    # ------------HORIZONTAL-LINE-DETECTION-------------#
    ####################################################

    # def detect2(self, output_filename=None):
    #     min_line_length = 20
    #     while True:
    #         ret, frame = self.cap.read()
    #
    #         self.height = frame.shape[0]
    #         self.width = frame.shape[1]
    #
    #         canny_image = canny(frame)
    #         warped_image = warp(canny_image, self.height, self.width)
    #         masked_image = apply_mask(warped_image, self.stencil)
    #         self.lines = cv2.HoughLinesP(masked_image, 2, np.pi / 180, 20, np.array([]), minLineLength=min_line_length)
    #
    #         is_turning = self.is_turning()
    #         horizontal_lines, detected_bool, detect_intensity = self.detect_horizontal(masked_image)
    #
    #         if is_turning == 1:
    #             print("right turn")
    #         elif is_turning == 2:
    #             print("left turn")
    #         if detected_bool:
    #             print(f"DETECTED, intensity: {detect_intensity}")
    #         display_lines_on_img(warped_image, horizontal_lines, thickness=10, wait=False)
    #         # cv2.imshow('tt', combo_image)
    #
    #         if cv2.waitKey(25) & 0xFF == ord('q'):
    #             break

    # def detect_horizontal(self, slope_threshold=0.02, screen_fractions=5):
    #     detected = []
    #     det_bool = False
    #     detection_dist_intensity = -1  # 1 to 3 metric of how close is the detected line
    #     if self.lines is None:
    #         return np.array(detected), det_bool, detection_dist_intensity
    #     if self.is_turning() < 0:
    #         for line in self.lines:
    #             # print(line)
    #             # self.calculate_line_slope(line)
    #             # print(self.calculate_line_slope(line))
    #             # print(self.calculate_line_lenght(line)[0])
    #             x1, y1, x2, y2 = line[0]
    #
    #             # self.display_lines_on_img(image, line, thickness=10)
    #             if calculate_line_lenght(line)[0] > self.width / 5:  # megalytero apo to 1/5 ths eikonas
    #                 if abs(calculate_line_slope(line)[0]) < slope_threshold:
    #                     detected.append(line)
    #     avg_y = -1
    #     if len(detected) > 0:
    #         sum = 0
    #         det_bool = True
    #         for d in detected:
    #             sum += d[0][1]
    #         avg_y = sum / len(detected)
    #         fraction = avg_y / self.height
    #         detection_dist_intensity = round(fraction * screen_fractions + 1, None)
    #         # if avg >= self.height / 3:
    #         #     detection_dist_intensity = 1
    #         # elif self.height / 3 > avg >= 2 * self.height / 3:
    #         #     detection_dist_intensity = 2
    #         # else:
    #         #     detection_dist_intensity = 3
    #
    #     return np.array(detected), det_bool, detection_dist_intensity, avg_y


if __name__ == "main":
    # lk = Lane_detection("real_tests_picam/straight_line.mp4", use_mask_class=True, mask_filename="raspi_test.json")
    # lk = Lane_detection("real_tests_picam/straight _ turn.mp4", use_mask_class=True, mask_filename="raspi_test.json")
    # lk = Lane_detection("real_tests_picam/random.mp4", use_mask_class=True, mask_filename="raspi_test.json")
    lk = Detection("real_tests_picam/straight_line _ roundabout.mp4", mask_filename="lane_det_test_mask.json")
    lk.detect()
