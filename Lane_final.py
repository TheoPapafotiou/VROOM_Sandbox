import cv2
import numpy as np
from numpy.core.fromnumeric import size
from numpy.lib.function_base import average


def canny(image):
    # Returns the processed image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 100, 300)
    return canny


def masked_region(image, height, width):
    # mask over the whole frame
    polygons = np.array([
        [(0, height), (width / 4, height / 4), (width * 3 / 4, height / 4), (width, height)]  # (y,x)
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, np.int32([polygons]), 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def make_stencil(image, height, width):
    # mask over the whole frame
    polygons = np.array([
        [(0, height), (width / 4, height / 4), (width * 3 / 4, height / 4), (width, height)]  # (y,x)
    ])
    canny_image = canny(image)
    stencill = np.zeros_like(canny_image)
    cv2.fillPoly(stencill, np.int32([polygons]), 255)
    return stencill


def apply_mask(image, stencill):
    masked_imagee = cv2.bitwise_and(image, stencill)
    return masked_imagee


def warp(image, height, width):
    # Transforms the image to bird-view(-ish)

    # Destination points for warping
    dst_points = np.float32([[0, height], [width, height], [0, 0], [width, 0]])
    src_points = np.float32([[0, height], [width, height], [width / 4, height / 4], [width * 3 / 4, height / 4]])

    warp_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    inv_warp_matrix = cv2.getPerspectiveTransform(dst_points, src_points)

    warped_frame = cv2.warpPerspective(image, warp_matrix, (width, height))
    return warped_frame


def make_coordinates(line_parameters, max_y):
    x1, x2, y1, y2 = 0, 0, 0, 0
    if len(line_parameters) != 0:
        slope, intercept = line_parameters

    y1 = max_y
    y2 = y1 * (3 / 5)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])




# def get_horizontal_lanes(lines):
#     for line in lines:
#         x1,y1,x2,y2=line[0]
#     return horizontal_lines
def make_average_lines_2(lines, width, image):
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
        display_lines(lined_image, l)

    cv2.imshow("ttttt",lined_image)
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
        left_line = make_coordinates(left_fit_average, end_y)
    if len(right_fit) > 0:
        right_line = make_coordinates(right_fit_average, end_y)

    return (left_line, right_line), info


def make_average_lines(lines, width):
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
    info["avg_total_apoklisi"] = round(apoklisi/(2*len(lines)), 3)
    info["avg_kept_apoklisi"] = round(kept_apoklisi/(2*len(lines)-count), 3)
    info["not_kept"] = round(count/2, 3)
    # if len(horizontal_lines > 0):
    #     get_horizontal_lanes(horizontal_lines)

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    if len(left_fit) > 0:
        left_line = make_coordinates(left_fit_average, end_y)
    if len(right_fit) > 0:
        right_line = make_coordinates(right_fit_average, end_y)

    return (left_line, right_line), info


def display_lines(image, av_lines):
    line_image = np.zeros_like(image)
    if av_lines is not None:
        for line in av_lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 10)
    return line_image


cap = cv2.VideoCapture("test_videos/cam_test.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
ret, fframe = cap.read()
stencil = make_stencil(fframe, fframe.shape[0], fframe.shape[1])
out = cv2.VideoWriter('output_videos/output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20,
                      (frame_width, frame_height), True)

# Importing the PIL library
from PIL import Image
from PIL import ImageDraw

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    height = frame.shape[0]
    width = frame.shape[1]

    canny_image = canny(frame)
    wrapped_image = warp(canny_image, height, width)
    # cv2.imshow('tt', wrapped_image)
    # cv2.w aitKey()
    # cv2.imshow('trt', canny_image)
    # cv2.waitKey()
    # masked_image=masked_region(wrapped_image,height,width)
    masked_image = apply_mask(wrapped_image, stencil)
    lines = cv2.HoughLinesP(wrapped_image, 2, np.pi / 180, 20, np.array([]), minLineLength=5, maxLineGap=5)

    averaged_lines, info = make_average_lines(lines, width)

    # test, testinfo = make_average_lines_2(lines, width, wrapped_image)
    lined_image = display_lines(wrapped_image, averaged_lines)
    combo_image = cv2.addWeighted(wrapped_image, 0.8, lined_image, 1, 1)

    # # Open an Image
    # img = Image.open('car.png')
    # Call draw Method to add 2D graphics in an image
    # I1 = ImageDraw.Draw(combo_image)
    # image = Image.fromarray(combo_image)
    # I1 = ImageDraw.Draw(image)
    # # Add Text to an image
    # I1.text((28, 36), f"lines found:    {info['num_of_lines']}", fill=(255,0,0))
    # # I1.text((28, 56), f"avg tot err:    {info['avg_total_apoklisi']}", fill=(255, 0, 0))
    # # I1.text((28, 76), f"avg kept err:   {info['avg_kept_apoklisi']}", fill=(255, 0, 0))
    # # I1.text((28, 96), f"not kept lines: {info['not_kept']}", fill=(255, 0, 0))

    print(info)

    # # Open an Image
    # img = Image.open('car.png')
    # Call draw Method to add 2D graphics in an image
    # I1 = ImageDraw.Draw(combo_image)
    # image = Image.fromarray(combo_image)
    # I1 = ImageDraw.Draw(image)
    # # Add Text to an image
    # I1.text((28, 36), f"lines found:    {info['num_of_lines']}", fill=(255,0,0))
    # # I1.text((28, 56), f"avg tot err:    {info['avg_total_apoklisi']}", fill=(255, 0, 0))
    # # I1.text((28, 76), f"avg kept err:   {info['avg_kept_apoklisi']}", fill=(255, 0, 0))
    # # I1.text((28, 96), f"not kept lines: {info['not_kept']}", fill=(255, 0, 0))

    print(info)

    # # Open an Image
    # img = Image.open('car.png')
    # Call draw Method to add 2D graphics in an image
    # I1 = ImageDraw.Draw(combo_image)
    # image = Image.fromarray(combo_image)
    # I1 = ImageDraw.Draw(image)
    # # Add Text to an image
    # I1.text((28, 36), f"lines found:    {info['num_of_lines']}", fill=(255,0,0))
    # # I1.text((28, 56), f"avg tot err:    {info['avg_total_apoklisi']}", fill=(255, 0, 0))
    # # I1.text((28, 76), f"avg kept err:   {info['avg_kept_apoklisi']}", fill=(255, 0, 0))
    # # I1.text((28, 96), f"not kept lines: {info['not_kept']}", fill=(255, 0, 0))

    print(info)

    # out.write(combo_image)
    # image.show()
    cv2.imshow('tt', combo_image)
    cv2.waitKey()
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cap.release()
out.release()
