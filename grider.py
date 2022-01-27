import cv2

### Change below ###

frame = cv2.imread("Merged_v2_lab.jpg")
size_of_track_w = 490 #cm
size_of_track_h = 370 #cm
cm_per_square = 10

### Change above ###

height, width, _ = frame.shape
print(height, width)

interval_x = cm_per_square*(width)/size_of_track_w
interval_y = cm_per_square*(height)/size_of_track_h

for k in range (int(width), 0, -int(interval_x)):
    cv2.line(frame, (k, 0), (k, int(height)), (255, 0, 255), 2, 1)

for p in range (int(height), 0, -int(interval_y)):
    cv2.line(frame, (0, p), (int(width), p), (255, 0, 255), 2, 1)

cv2.circle(frame, (int(width-(0*interval_x)), int(height-(0*interval_y))), radius=20, color=(0, 255, 255), thickness=-1)
        
cv2.imwrite("Track_v2_IRL_" + str(cm_per_square) + "cm.png", frame)