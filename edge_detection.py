# edge detection 
import cv2
import matplotlib.pyplot as plt 
import numpy as np
from numpy.core.overrides import verify_matching_signatures


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # reduce noise  -> filter with guassian filter
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #canny method to find lanes -> based on colour
    canny = cv2.Canny(blur, 50, 150)
    return canny

'''
First Way (+)
image = plt.imread('kernel_test.png')
plt.imshow(image)

vertical_filter = [[-1, 2, -1],[0, 0, 0],[1, 2, 1]]

horizontal_filter =[[-1, 0, -1],[-2, 0, 2],[-1, 0, 1]]

n,m,d = image.shape


edges_img = np.zeros_like(image)

for row in range(3,n-2):
    for col in range(3,m-2): #i chose to ignore the edges of the img because there is not enough pixels to apply my 3x3 arrayfilter
        local_pixels = image[row-1:row+2, col-1:col+2, 0] 

        v_trasformed_pixels = vertical_filter*local_pixels
        vertical_score = v_trasformed_pixels.sum() + 4 #I can take values between -4,4 so normilization 
        # zero to one range

        h_trasformed_pixels = horizontal_filter * local_pixels
        horizontal_score = h_trasformed_pixels.sum()/4

        edge_score = (vertical_score**2 + horizontal_score**2)**.5
        edges_img[row,col] = [edge_score]*3

edges_img = edges_img/edges_img.max()
plt.imshow(edges_img)
'''
'''
SECOND WAY (++)
img = cv22.imread("kernel_test.png")
cv22.imshow('img', img)
gray = cv22.cv2tColor(img, cv22.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv22.cornerHarris(gray, 2,3,0.04)

dst = cv22.dilate(dst,None)

img[dst>0.01*dst.max()] = [0, 0, 255]
cv22.imshow('dst', img)

if cv22.waitKey(0) & 0xff == 27:
    cv22.destroyAllWindows()

    '''
'''
# Third way - (+++)

from matplotlib import pyplot as plt
def canny(image):
    gray = cv22.cv2tColor(image, cv22.COLOR_RGB2GRAY)
    # reduce noise  -> filter with guassian filter
    blur = cv22.GaussianBlur(gray,(5,5),0)
    #canny method to find lanes -> based on colour
    canny = cv22.Canny(blur, 50, 150)
    return canny
  
def region_of_interest(image,option):
    #mask
    height = image.shape[0]
    width = image.shape[1]
    #for options == 1 when the car moves straight
    if option == 1:
        polygons=np.array([
        [(0,height),(int(width/2.5),int(height/3.5)),(int(width/1.6),int(height/3.5)),(width,height)] #(y,x)
        ])
    elif option == 2: #options == 2 when the car turn right small
        polygons=np.array([
        [(int(width/1.65), int(height/1.68)),(width,int(height/4.1)),(width,height),(int(width/1.65),height)] #(y,x)
        ])
    elif option == 3:
        polygons=np.array([
        [(1,1),(650,350),(1030,350),(width,height)] #(y,x)
        ])

    # apply it to a block mask
    mask = np.zeros_like(image)
    cv22.fillPoly(mask, polygons, 255)
    masked_image = cv22.bitwise_and(image,mask)
    return masked_image
    
x_points = []
y_points = []

cap = cv22.VideoCapture("straight_simulation.mp4")
while (cap.isOpened()):
    _,frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image,1)
    #cv22.imshow("result", cropped_image)
    width  = cap.get(cv22.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = cap.get(cv22.CAP_PROP_FRAME_HEIGHT)  # float `height`
    corners = cv22.goodFeaturesToTrack(cropped_image, 27, 0.01, 10)
    corners = np.int0(corners)

  
    # we iterate through each corner, 
    # making a circle at each point that we think is a corner.

    for i in corners:
        x, y = i.ravel()
        x_points.append(x)
        y_points.append(y)
        cv22.circle(cropped_image, (x, y), 3, 255, -1)
    white = (255,255,255)
    
    cv22.line(cropped_image, (int(x_points[2]),int(y_points[2])), (50,int(height)), white, 15)
    cv22.imshow("result", cropped_image)

    if cv22.waitKey(5) == ord('q'):
        break
cap.release()
cv22.destroyAllWindows()
'''

# tetartos tropos -> somehow operating 

from matplotlib import pyplot as plt
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # reduce noise  -> filter with guassian filter
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #canny method to find lanes -> based on colour
    canny = cv2.Canny(blur, 50, 150)
    return canny
  
def region_of_interest(image,option):
    #mask
    height = image.shape[0]
    width = image.shape[1]
    #for options == 1 when the car moves straight
    if option == 1:
        polygons=np.array([
        [(0,height),(int(width/2.5),int(height/3.5)),(int(width/1.6),int(height/3.5)),(width,height)] #(y,x)
        ])
    elif option == 2: #options == 2 when the car turn right small
        polygons=np.array([
        [(int(width/1.65), int(height/1.68)),(width,int(height/4.1)),(width,height),(int(width/1.65),height)] #(y,x)
        ])
    elif option == 3:
        polygons=np.array([
        [(1,1),(650,350),(1030,350),(width,height)] #(y,x)
        ])

    # apply it to a block mask
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image
def slope(x1,y1,x2,y2):
    if x2!=x1:
        return((y2-y1)/(x2-x1))
    else:
        return 'NA'
def drawLine(img, x1,y1,x2,y2):
    m = slope(x1,y1,x2,y2)
    h,w = img.shape[: 2]
    if m != 'NA':

        # starting point
        px = 0
        py = -(x1-0)*m + y1
        # ending point
        qx = w
        qy = -(x2-w)*m + y2
        cv2.line(img,(int(px), int(py)), (int(qx),int(qy)), (255,255,255), 12)

    
x_points = []
y_points = []

cap = cv2.VideoCapture("straight_simulation.mp4")
<<<<<<< HEAD

out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))

=======
out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))


>>>>>>> e0a1c20b7130a22a4dedd6024d951e2c70f76a0e
while (cap.isOpened()):
    _,frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image,1)
<<<<<<< HEAD
    out.write(cropped_image)

    #cv2.imshow("result", cropped_image)
=======
    #cv2.imshow("result", cropped_image)
    out.write(cropped_image)

>>>>>>> e0a1c20b7130a22a4dedd6024d951e2c70f76a0e
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    corners = cv2.goodFeaturesToTrack(cropped_image, 27, 0.01, 10)
    corners = np.int0(corners)
  
    # we iterate through each corner, 
    # making a circle at each point that we think is a corner.

    for i in corners:
        x, y = i.ravel()
        x_points.append(x)
        y_points.append(y)
        cv2.circle(cropped_image, (x, y), 3, 255, -1)

#    print(y_points[2])
    white = (255,255,255)
    
    #drawLine(cropped_image,50,int(height),544,233)
    cv2.line(cropped_image, (int(x_points[2]),int(y_points[2])), (50,int(height)), white, 15)
    cv2.imshow("result", cropped_image)

    if cv2.waitKey(5) == ord('q'):
        break
out.release()
cv2.destroyAllWindows()



'''
img = cv2.imread('straight_2.png')
cv2.namedWindow('Harris Corner Detection Test', cv2.WINDOW_NORMAL)

def f(x=None):
    return

cv2.createTrackbar('Harris Window Size', 'Harris Corner Detection Test', 5, 25, f)
cv2.createTrackbar('Harris Parameter', 'Harris Corner Detection Test', 1, 100, f)
cv2.createTrackbar('Sobel Aperture', 'Harris Corner Detection Test', 1, 14, f)
cv2.createTrackbar('Detection Threshold', 'Harris Corner Detection Test', 1, 100, f)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

img_bak = img

while True:
    img = img_bak.copy()

    window_size = cv2.getTrackbarPos('Harris Window Size', 'Harris Corner Detection Test')
    harris_parameter = cv2.getTrackbarPos('Harris Parameter', 'Harris Corner Detection Test')
    sobel_aperture = cv2.getTrackbarPos('Sobel Aperture', 'Harris Corner Detection Test')
    threshold = cv2.getTrackbarPos('Detection Threshold', 'Harris Corner Detection Test')

    sobel_aperture = sobel_aperture * 2 + 1

    if window_size <= 0:
        window_size = 1

    dst = cv2.cornerHarris(gray, window_size, sobel_aperture, harris_parameter/100)

    # Threshold for an optimal value, it may vary depending on the image.
    _ , dst_thresh = cv2.threshold(dst, threshold/100 * dst.max(), 255, 0)
    dst_thresh = np.uint8(dst_thresh)

    dst_show = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    dst_show = np.uint8(dst_show)

    ## REFINE CORNERS HERE!

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst_thresh)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5,5), (-1,-1), criteria)

    try:
        # Now draw them
        corners = np.int0(corners)

        img[corners[:, 1], corners[:, 0]] = [0, 255, 0]
        img[dst_thresh > 1] = [0, 0, 255]

    except:
        pass

    cv2.imshow('Harris Corner Detection Test', np.hstack((img, dst_show)))

    if cv2.waitKey(10) & 0xFF == 27:
        break

cv2.destroyAllWindows()
'''
'''
import cv2
import numpy as np

img_file = 'straight_1.png'
img = cv2.imread(img_file, cv2.IMREAD_COLOR)

imgDim = img.shape
dimA = imgDim[0]
dimB = imgDim[1]

# RGB to Gray scale conversion
img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# Noise removal with iterative bilateral filter(removes noise while preserving edges)
noise_removal = cv2.bilateralFilter(img_gray,9,75,75)
# Thresholding the image
ret,thresh_image = cv2.threshold(noise_removal,220,255,cv2.THRESH_OTSU)
th = cv2.adaptiveThreshold(noise_removal, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Applying Canny Edge detection
canny_image = cv2.Canny(th,250,255)
canny_image = cv2.convertScaleAbs(canny_image)

# dilation to strengthen the edges
kernel = np.ones((3,3), np.uint8)
# Creating the kernel for dilation
dilated_image = cv2.dilate(canny_image,kernel,iterations=1)
np.set_printoptions(threshold=np.nan)

_, contours, h = cv2.findContours(dilated_image, 1, 2)
contours= sorted(contours, key = cv2.contourArea, reverse = True)[:1]


corners    = cv2.goodFeaturesToTrack(thresh_image,6,0.06,25)
corners    = np.float32(corners)

for item in corners:
    x,y    = item[0]
    cv2.circle(img,(x,y),10,255,-1)
cv2.namedWindow("Corners", cv2.WINDOW_NORMAL)
cv2.imshow("Corners",img)
cv2.waitKey()
'''
'''
import numpy as np
import cv2 as cv

filename = 'straight_1.png'
img = cv.imread(filename)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)
#result is dilated for marking the corners, not important
dst = cv.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
cv.imshow('dst',img)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
'''
'''
from matplotlib import pyplot as plt

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # reduce noise  -> filter with guassian filter
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #canny method to find lanes -> based on colour
    canny = cv2.Canny(blur, 50, 150)
    return canny
  
def region_of_interest(image,option):
    #mask
    height = image.shape[0]
    width = image.shape[1]
    #for options == 1 when the car moves straight
    if option == 1:
        polygons=np.array([
        [(0,height),(int(width/2.5),int(height/3.5)),(int(width/1.6),int(height/3.5)),(width,height)] #(y,x)
        ])
    elif option == 2: #options == 2 when the car turn right small
        polygons=np.array([
        [(int(width/1.65), int(height/1.68)),(width,int(height/4.1)),(width,height),(int(width/1.65),height)] #(y,x)
        ])
    elif option == 3:
        polygons=np.array([
        [(1,1),(650,350),(1030,350),(width,height)] #(y,x)
        ])

    # apply it to a block mask
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image
def slope(x1,y1,x2,y2):
    if x2!=x1:
        return((y2-y1)/(x2-x1))
    else:
        return 'NA'
def drawLine(img, x1,y1,x2,y2):
    m = slope(x1,y1,x2,y2)
    h,w = img.shape[: 2]
    if m != 'NA':

        # starting point
        px = 0
        py = -(x1-0)*m + y1
        # ending point
        qx = w
        qy = -(x2-w)*m + y2
        cv2.line(img,(int(px), int(py)), (int(qx),int(qy)), (255,255,255), 12)

    
x_points = []
y_points = []

cap = cv2.VideoCapture("straight_simulation.mp4")
while (cap.isOpened()):
    _,frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image,1)
    #cv2.imshow("result", cropped_image)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    corners = cv2.goodFeaturesToTrack(cropped_image,3,13,0.05)
    kernel = np.ones((7,7),np.uint8)
    corners = cv2.dilate(corners, kernel,iterations = 2)
    cropped_image[corners>0.025 * corners.max()] = [255,127,127]

    for i in corners:
        x, y = i.ravel()
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    for i in corners:
        x, y = i.ravel()
        x_points.append(x)
        y_points.append(y)
        cv2.circle(cropped_image, (x, y), 3, 255, -1)

#    print(y_points[2])
    white = (255,255,255)
    
    #drawLine(cropped_image,50,int(height),544,233)
    cv2.line(cropped_image, (int(x_points[2]),int(y_points[2])), (50,int(height)), white, 15)
    cv2.imshow("result", cropped_image)

    if cv2.waitKey(5) == ord('q'):
        break
cv2.destroyAllWindows()
<<<<<<< HEAD
'''
=======
'''
>>>>>>> e0a1c20b7130a22a4dedd6024d951e2c70f76a0e
