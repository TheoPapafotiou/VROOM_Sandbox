import cv2
import numpy as np
import math

from Mask import Mask


def canny(image):
    # Returns the processed image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 100, 300)
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


class DetectHorizontal:
    height = 0
    width = 0
    lines = np.empty((1, 1, 1))

    def __init__(self, mask_filename="default_mask.json"):
        self.mask = Mask(filename=mask_filename)
        self.stencil = self.mask.stencil

    def detection(self, image, min_line_length=200):
        self.height = image.shape[0]
        self.width = image.shape[1]
        # masked_image = self.mask.apply_mask(input_image=canny(image))
        masked_image = self.mask.apply_mask(image)
        self.lines = cv2.HoughLinesP(image, 2, np.pi / 180, 20, np.array([]), minLineLength=min_line_length)

        detected_lines, info_dict = self.detect_horizontal()
        display_lines_on_img(image, detected_lines)
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

    def detect_horizontal(self, slope_threshold=0.02, screen_fractions=5):
        info_dict = {
            "det_bool": False,
            "detection_dist_intensity": -1,  # 1 to <screen_fractions> metric of how close is the detected line
            "avg_y": -1,
            "min_y": -1,
        }
        detected = []
        detection_dist_intensity = -1  # 1 to 3 metric of how close is the detected line
        if self.lines is None:
            return np.array(detected), info_dict
        if self.is_turning() < 0:
            for line in self.lines:
                # print(line)
                # self.calculate_line_slope(line)
                # print(self.calculate_line_slope(line))
                # print(self.calculate_line_lenght(line)[0])
                x1, y1, x2, y2 = line[0]

                # self.display_lines_on_img(image, line, thickness=10)
                if calculate_line_length(line)[0] > self.width / 5:  # megalytero apo to 1/5 ths eikonas
                    if abs(calculate_line_slope(line)[0]) < slope_threshold:
                        detected.append(line)
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
        info_dict["detection_dist_intensity"] = detection_dist_intensity
        info_dict["avg_y"] = avg_y
        info_dict["det_bool"] = det_bool

        return np.array(detected), info_dict




frame = cv2.imread('frame3383.png',0)
det = DetectHorizontal()
print(det.detection(frame))


