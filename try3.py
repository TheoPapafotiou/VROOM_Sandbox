from this import d
import numpy as np
import cv2
import time

from matplotlib import pyplot as plt
import matplotlib as mpl

from sign_detection import SignDetection

from sys import platform

if platform == "darwin":
	mpl.use('macosx')

image_list = []


def get_image_list(list):
	imgs = []
	for i in list:
		imgs.append(np.load(f"depth2/d/RealSense_{i}.npy"))


def get_d_image(filename):
	return np.load(filename)


def get_rgb_image(filename):
	return np.asanyarray(cv2.imread(filename))


def do_colormap(image):
	a = (255 / image.max())
	image_clrmp = cv2.convertScaleAbs(image, alpha=a)
	image_clrmp = cv2.applyColorMap(image_clrmp, cv2.COLORMAP_HSV)
	return image_clrmp


def point_finding(sample_img, plot_title="click to copy the point"):
	import pyperclip
	fig, ax = plt.subplots()
	plt.title(plot_title)
	ax.imshow(sample_img)

	def on_click(event):
		# print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
		#       ('double' if event.dblclick else 'single', event.button,
		#        e  vent.x, event.y, event.xdata, event.ydata))
		print(
			f"[{int(round(event.xdata, 0))},{int(round(event.ydata, 0))}], {sample_img[int(round(event.ydata, 0))][int(round(event.xdata, 0))]}")
		pyperclip.copy(f"[{int(round(event.xdata, 0))},{int(round(event.ydata, 0))}]")

	fig.canvas.mpl_connect('button_press_event', on_click)
	plt.show()


## testings

# rgb = get_rgb_image("depth2/rgb/RealSense_color_1.jpg")
#
# cv2.imshow("", rgb)
# cv2.waitKey()

# sd = SignDetection()
#
# color_frame = get_rgb_image("depth2/rgb/RealSense_try2_color_10.jpg")
# color_image = np.asanyarray(color_frame)
#
# result, data = sd.detectSign(color_image, 480, 640)
# print(result)
# print(data)

class Vehicle_detection:
	sd = SignDetection()

	def __init__(self):
		pass

	def detect(self):
		color_image = get_rgb_image("depth2/rgb/RealSense_try2_color_10.jpg")
		depth_image = get_d_image("depth2/d/RealSense_try2_10.npy")

		depth_colormap = do_colormap(depth_image)

		res, _ = self.sd.detectSign(color_image, 480, 640)
		# print(res)
		data = res["Box"]

		x = data["x"]
		y = data["y"]
		w = data["width"]
		h = data["height"]


		depth_clmp_copy = np.copy(depth_colormap)
		cv2.rectangle(depth_clmp_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)

		cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

		cv2.imshow("clr", color_image)
		cv2.waitKey()
		#
		cv2.imshow("dpth",depth_colormap)
		cv2.waitKey()

		# print(result)
		print(data)

		# get the center pixels on the depth camera to see the distance of the object

		# point_finding(depth_image)

		# 6x6 center matrix
		distances = depth_image[int(y + h / 2) - 3: int(y + h / 2) + 3, int(x + w / 2) - 3:int(x + w / 2) + 3]
		distance = int(np.mean(distances))

		distances_excl = distances[abs(distances-distance) < 2]

		print(distances)

		print(distance)

		print(distances_excl)

		depth_image[depth_image < distance - 100] = 0
		# thresh_clrmp = do_colormap(depth_image)
		# cv2.imshow("thresh", depth_image)
		# cv2.waitKey()

		depth_image[depth_image > distance + 100] = 0
		# thresh_clrmp = do_colormap(depth_image)
		# cv2.imshow("thresh", depth_image)
		# cv2.waitKey()

		# convert to 8-bit image
		a = (255 / depth_image.max())
		depth_image = cv2.convertScaleAbs(depth_image, alpha=a)
		ss = np.mean(depth_image)

		## mask the other area
		mask = np.zeros(( 480, 640), dtype="uint8")
		cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
		depth_image = cv2.bitwise_and(depth_image, depth_image, mask=mask)
		# cv2.imshow("masked", depth_image)
		# cv2.waitKey()

		blur = cv2.GaussianBlur(depth_image, (5, 5), 2)
		canny = cv2.Canny(blur, 100, 200,apertureSize=3)
		cv2.imshow("canny", canny)
		cv2.waitKey()
		contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		cv2.drawContours(image=depth_colormap, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
						 lineType=cv2.LINE_AA)

		cv2.imshow("final", depth_colormap)
		cv2.waitKey()

		# depth_image_fin = cv2.addWeighted(depth_colormap[:2], 0.5, canny, 0.5, 0.0)
		# cv2.imshow("finale", depth_image_fin)
		# cv2.waitKey()





vd = Vehicle_detection()
vd.detect()
