## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import time

from sign_detection import SignDetection

print("waits 6s to initialize DNN")
sd = SignDetection() # ~ 6 secs delay due to initializations
time.sleep(6)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
	if s.get_info(rs.camera_info.name) == 'RGB Camera':
		found_rgb = True
		break
if not found_rgb:
	print("The demo requires Depth camera with Color sensor")
	exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
	config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
	config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

try:
	count = 0
	# time.sleep(5)
	while True:

		input("press key to take a picture")

		# Wait for a coherent pair of frames: depth and color
		frames = pipeline.wait_for_frames()

		# ---- ALIGNED ----
		# align depth frame to color frame
		aligned_frames = align.process(frames)
		# Get aligned frames
		aligned_depth_frame = aligned_frames.get_depth_frame()
		aligned_color_frame = aligned_frames.get_color_frame()

		# Convert images to numpy arrays
		# aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
		aligned_color_image = np.asanyarray(aligned_color_frame.get_data())
		# print(aligned_depth_image.dtype)

		color_image = aligned_color_image
		color_img_dim = color_image.shape

		data1 , data2 = sd.detectSign(color_image.copy(), color_img_dim[0], color_img_dim[1])
		label = data1['Label']


		# Show images
		# cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
		# cv2.imshow('RealSense', images)
		# cv2.imwrite('RealSense1_' + str(count) + '.jpg', images1)
		# cv2.imwrite('RealSense2_' + str(count) + '.jpg', depth_image)

		## saving as .npy
		# cv2.imwrite('RealSense_color_' + str(count) + '.jpg', color_image)

		filename = f"dataset/{label}_{count}_{time.strftime('%H-%M-%S', time.gmtime())}.jpg"

		cv2.imwrite(filename, color_image)
		# cv2.imshow('RealSense', color_image)


		


		# if cv2.waitKey(1) & 0xFF == ord('q'):
		#     break
		print(f"{label}, {count}")
		# time.sleep(3)
		count += 1

finally:

	# Stop streaming
	pipeline.stop()