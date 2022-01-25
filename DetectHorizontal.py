import cv2
import numpy as np
import math

from Mask import Mask


def canny(image):
    # Returns the processed image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # cv2.imshow("gray", gray)
    # cv2.waitKey()
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    # cv2.imshow("blur", blur)
    # cv2.waitKey()
    canny = cv2.Canny(blur, 100, 200)
    # cv2.imshow("canny", canny)
    # cv2.waitKey()
    return canny


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


def calculate_line_length(lines):
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


def display_lines_on_img2(img, lines, thickness=10, wait=True, info_dict=None):
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
        if len(lines) != 1 or len(lines) != 0:
            for line in lines:
                # x1, y1, x2, y2 = line.reshape(4)
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), thickness=thickness)
                combined_image = cv2.addWeighted(img, 0.8, line_image, 1, 1)
                if info_dict is not None:
                    margin = 0
                    for key in info_dict:
                        cv2.putText(combined_image, f"{key} : {info_dict[key]}", (50, 50 + margin),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
                        margin += 40

                cv2.imshow("lines", combined_image)
                if wait:
                    cv2.waitKey()
            # if wait:
            #     cv2.waitKey()
        else:
            x1, y1, x2, y2 = lines[0]
            cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), thickness=thickness)
            combined_image = cv2.addWeighted(img, 0.8, line_image, 1, 1)
            cv2.imshow("lines", combined_image)
            if wait:
                cv2.waitKey()
    except:
        cv2.imshow("lines", img)
        if wait:
            cv2.waitKey()


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
                # if wait:
                #     cv2.waitKey()
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


def make_stencil(polygon, height, width):
    stencil = np.zeros((height, width), dtype=np.uint8)
    # cv2.fillPoly(stencil, np.int32([polygon]), 255)
    poly = np.array(polygon, dtype=np.int64)
    cv2.fillConvexPoly(stencil, poly, color=(255, 255, 255))
    return stencil


def apply_mask(image, stencill):
    masked = cv2.bitwise_and(image,
                             image,
                             mask=stencill)
    return masked


class DetectHorizontal:
    height = 0
    width = 0
    lines = np.empty((1, 1, 1))
    precision_state = False

    def __init__(self, mask_filename="default_mask_real.json", sample_img=None):
        self.mask = Mask(filename=mask_filename, sample_img=sample_img)
        # self.precision_mask = Mask(filename=f"precision_{mask_filename}", sample_img=sample_img)
        self.shape = self.mask.mask["shape"]
        self.height = self.shape[0]
        self.width = self.shape[1]
        self.precision_stencil = self.make_precision_stencil()
        self.stencil = self.mask.stencil

    def detection(self, image, stop_signal_at=300, min_line_length=100, reset=False, precision_state_enabled = False):
        # self.height = image.shape[0]
        # self.width = image.shape[1]
        # masked_image = self.mask.apply_mask(input_image=canny(image))
        canny_img = canny(image)
        if reset:
            self.precision_state = False
        if self.precision_state is False:
            masked_image = self.mask.apply_mask(canny_img)
        else:
            # masked_image = self.precision_mask.apply_mask(canny_img)
            # print()
            masked_image = apply_mask(canny_img, self.precision_stencil)
        # self.lines = cv2.HoughLinesP(masked_image, 5, np.pi / 180, 150, np.array([]), minLineLength=min_line_length)
        self.lines = cv2.HoughLinesP(masked_image, 1, np.pi / 180, 50, np.array([]),
                                     minLineLength=min_line_length, maxLineGap=100)
        detected_lines, info_dict = self.detect_horizontal(slope_threshold=0.05)
        if precision_state_enabled:
            if 0 < info_dict["detection_dist_intensity"] < 4 and self.precision_state is False:
                self.precision_state = True
                print("precision_state")
            if info_dict["avg_y"] == -1 and self.precision_state is True:
                self.precision_state = False
                print("exited precision_state")
        if int(info_dict["min_y"]) > (self.height - stop_signal_at):
            info_dict["stop_signal"] = 1
        print(info_dict)
        # display_lines_on_img2(image, detected_lines, wait=True, info_dict=info_dict)
        # display_lines_on_img(image, detected_lines, wait=True)

        return info_dict

    def is_turning(self):
        slopes = calculate_line_slope(self.lines)
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

    def make_precision_stencil(self, fraction_precision_mask=3.5
                               ):
        height = self.height
        fraction_height = self.height - self.height / fraction_precision_mask
        width = self.width
        polygon = [[0, height - 1], [0, fraction_height - 1], [width, fraction_height - 1], [width - 1, height - 1]]

        stencil = make_stencil(polygon, height, width)
        return stencil
        # masked_image = apply_mask(canny_img, stencil)

    def detect_horizontal(self, slope_threshold=0.05, screen_fractions=12, image=None, threshold_avg_x=200,
                          x_avg_enabled=False,
                          non_y_avg_exclusion_enabled=True, avg_y_exclusion_threshold=50):
        info_dict = {
            "detection_dist_intensity": -1,  # 1 to <screen_fractions> metric of how close is the detected line
            "detected_boolean": False,
            "avg_y": -1,
            "min_y": -1,
            "lines_found": 0,
            "stop_signal": 0
        }
        detected = []
        detection_dist_intensity = -1  # 1 to 3 metric of how close is the detected line
        avg_y = -1
        min_y = -1

        if self.lines is None:
            return np.array(detected), info_dict
        if self.is_turning() < 0:
            for line in self.lines:
                # print(line)
                # self.calculate_line_slope(line)
                # print(self.calculate_line_slope(line))
                # print(self.calculate_line_lenght(line)[0])
                x1, y1, x2, y2 = line[0]

                # display_lines_on_img(image, line, thickness=10)

                # if calculate_line_length(line)[0] > self.width / 10 :  # megalytero apo to 1/5 ths eikonas
                #     print("passes length")

                if abs(calculate_line_slope(line)[0]) < slope_threshold:
                    detected.append(line)
        else:
            return np.array(detected), info_dict
        det_bool = False
        if len(detected) > 0:

            # for d in detected:
            #     average_y = (d[0][1] + d[0][3])/2
            #     length = calculate_line_length(d)
            #
            if x_avg_enabled:
                leftest_x = self.width
                rightest_x = 0
                sum_x = 0
                for d in detected:
                    if d[0][0] > rightest_x:
                        rightest_x = d[0][0]
                    if d[0][2] > rightest_x:
                        rightest_x = d[0][2]
                    if d[0][0] < leftest_x:
                        leftest_x = d[0][0]
                    if d[0][2] < leftest_x:
                        leftest_x = d[0][2]
                    sum_x += d[0][0] + d[0][2]
                    avg_x = sum_x / (2 * len(detected))
                    if abs(avg_x - (self.width / 2)) > threshold_avg_x:
                        return np.array(detected), info_dict

            sum = 0
            det_bool = True
            for d in detected:
                if d[0][1] > min_y:  # the lower in the picture the bigger the y
                    min_y = d[0][1]
                if d[0][3] > min_y:
                    min_y = d[0][3]
                sum += d[0][1]  # y1
                sum += d[0][3]  # y2
            avg_y = sum / (2 * len(detected))
            fraction = avg_y / self.height
            detection_dist_intensity = screen_fractions - round(fraction * screen_fractions + 1, None)
            # if avg >= self.height / 3:
            #     detection_dist_intensity = 1
            # elif self.height / 3 > avg >= 2 * self.height / 3:
            #     detection_dist_intensity = 2
            # else:
            #     detection_dist_intensity = 3

            if non_y_avg_exclusion_enabled:
                detected = [d for d in detected if abs(d[0][1] - avg_y) < avg_y_exclusion_threshold]

        info_dict["detection_dist_intensity"] = detection_dist_intensity
        info_dict["avg_y"] = avg_y
        info_dict["min_y"] = min_y
        info_dict["lines_found"] = len(detected)
        info_dict["detected_boolean"] = det_bool

        return np.array(detected), info_dict
