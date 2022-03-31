import pyrealsense2 as rs
import numpy as np
import cv2
import time
from sign_detection import SignDetection
from pedestrian_detection_jetson import PedestrianDetectionJetson

pedDet = PedestrianDetectionJetson() # ~ 6 secs delay due to initializations

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
count = 0
try:
    while count <= 20:
        count += 1
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        start = time.time()
        result_image = pedDet.detectPedestrian(color_image)
        print(time.time() - start)

        cv2.imwrite('RealSense_Ped_Detect_' + str(count) + '.jpg', result_image)

        time.sleep(0.2)

finally:

    # Stop streaming
    pipeline.stop()