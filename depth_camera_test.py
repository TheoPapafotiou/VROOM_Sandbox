## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import time

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
    time.sleep(5)
    while count < 500:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # ---- ALIGNED ----
        # align depth frame to color frame
        aligned_frames = align.process(frames)
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()

        # Convert images to numpy arrays
        aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
        aligned_color_image = np.asanyarray(aligned_color_frame.get_data())
        print(aligned_depth_image.dtype)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        aligned_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(aligned_depth_image, alpha=0.1), cv2.COLORMAP_JET)

        ## saving as .npy
        cv2.imwrite('RealSense_aligned_rgb_' + str(count) + '.jpg', aligned_color_image)
        np.save('RealSense_aligned_d_' + str(count) + '.npy', aligned_depth_image)
        cv2.imwrite('RealSense_aligned_d_clmp_' + str(count) + '.jpg', aligned_depth_colormap)


        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        print(depth_image.dtype)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            print(type(resized_color_image))
            print(type(depth_colormap))
            images1 = np.hstack((resized_color_image, depth_colormap))
        else:
            
            images1 = np.hstack((color_image, depth_colormap))

        # Show images
        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RealSense', images)
        # cv2.imwrite('RealSense1_' + str(count) + '.jpg', images1)
        # cv2.imwrite('RealSense2_' + str(count) + '.jpg', depth_image)

        ## saving as .npy
        cv2.imwrite('RealSense_color_' + str(count) + '.jpg', color_image)
        np.save('RealSense_' + str(count) + '.npy', depth_image)

        


        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        print(count)
        time.sleep(3)
        count += 1

finally:

    # Stop streaming
    pipeline.stop()