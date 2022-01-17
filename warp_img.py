from heapq import merge
import cv2
import numpy as np

img = cv2.imread('GPS3.jpg')
height, width = img.shape[:2]
mrg = 10

# heightD = height + 2*mrg
# widthD = width + 2*mrg
# img_back = np.zeros([heightD, widthD, 3],dtype=np.uint8)
# img_back.fill(255)
# print(img_back.shape)
# img_back[mrg : mrg + height, mrg : mrg + width] = img

input_pts = np.float32([[mrg, mrg], [mrg, height - mrg], [width - mrg, mrg]])
output_pts = np.float32([[0, 0], [0, height], [width, 0]])

transform = cv2.getAffineTransform(input_pts , output_pts)
dst = cv2.warpAffine(img, transform, (width, height))

out = cv2.hconcat([img, dst])
cv2.imshow('Preview', out)
cv2.waitKey(0)