import cv2 
import numpy as np

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
    
## to show the coordinates of an image:


def click_event(event,x,y,flags,params):
    #event -> left mouse click
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canny_image, str(x) + ',' + str(y), (x,y), font,1,(255,0,0), 2)
        cv2.imshow('image', canny_image)


# for reviewing the mask
""" 
image = cv2.imread("Lanes.png")
lane_image = np.copy(image)
canny_image = canny(lane_image)
cv2.imshow("result", region_of_interest(canny_image,2))
cv2.waitKey(0)  
cv2.destroyAllWindows()
"""
# for finding the coordinates of an image
""" 
cv2.imshow('image', canny_image)
cv2.setMouseCallback('image', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
 """

# for the video

cap = cv2.VideoCapture("take_yaw_-20.mp4")
while (cap.isOpened()):
    _,frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image,2)
    cv2.imshow("result", cropped_image)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    # or
    #width  = cap.get(3)  # float `width`
    #height = cap.get(4)  # float `height`
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
 