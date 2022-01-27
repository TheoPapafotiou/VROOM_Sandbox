from pickletools import uint8
import cv2
import numpy as np


img1 = cv2.imread("Preview_20_GPS1.jpg")
img2 = cv2.imread("Preview_20_GPS2.jpg")
img3 = cv2.imread("Preview_20_GPS3.jpg")

width = 640
height = 480
dim = (width, height)
  
# resize image
img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
img3 = cv2.resize(img3, dim, interpolation = cv2.INTER_AREA)

## ======= Image 1 =======
rb1 = (width-30, height)
lb1 = (10, height) 
lu1 = (10, 155)
ru1 = (width-30, 155)

cv2.circle(img1, rb1, 5, color = (255, 0, 0), thickness=-1)
cv2.circle(img1, lb1, 5, color = (255, 0, 0), thickness=-1)
cv2.circle(img1, ru1, 5, color = (255, 0, 0), thickness=-1)
cv2.circle(img1, lu1, 5, color = (255, 0, 0), thickness=-1)

crop_img1 = img1[lu1[1]:lb1[1], lu1[0]:ru1[0]]

# cv2.imshow("Preview1", crop_img1)

## ======= Image 3 =======
rb3 = (width, 437)
lb3 = (40, 437) 
lu3 = (40, 0)
ru3 = (width, 0)

cv2.circle(img3, (rb3), 5, color = (255, 0, 0), thickness=-1)
cv2.circle(img3, (lb3), 5, color = (255, 0, 0), thickness=-1)
cv2.circle(img3, (ru3), 5, color = (255, 0, 0), thickness=-1)
cv2.circle(img3, (lu3), 5, color = (255, 0, 0), thickness=-1)

crop_img3 = img3[lu3[1]:lb3[1], lu3[0]:ru3[0]]

# cv2.imshow("Preview3", crop_img3)

vconcat = cv2.vconcat([crop_img3, crop_img1])
print(vconcat.shape)
# cv2.imshow("Vconcat", vconcat)

## ======= Image 2 =======
img2 = cv2.rotate(img2, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

rb2 = (405, 640)
lb2 = (0, 640) 
lu2 = (0, 0)
ru2 = (405, 0)

cv2.circle(img2, (rb2), 5, color = (255, 0, 0), thickness=-1)
cv2.circle(img2, (lb2), 5, color = (255, 0, 0), thickness=-1)
cv2.circle(img2, (ru2), 5, color = (255, 0, 0), thickness=-1)
cv2.circle(img2, (lu2), 5, color = (255, 0, 0), thickness=-1)

crop_img2 = img2[lu2[1]:lb2[1], lu2[0]:ru2[0]]

back = np.zeros((vconcat.shape[0], crop_img2.shape[1], 3), dtype='uint8')
back[35:35+crop_img2.shape[0], 0:crop_img2.shape[1]] = crop_img2

final = cv2.hconcat([back, vconcat])

cv2.imshow("Final", final)
cv2.imwrite("Merged_v2_lab.jpg", final)
cv2.waitKey(0)

