# from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sys import platform
if platform == "darwin":
    mpl.use('macosx')


filename1 = "depth/RealSense_4.npy"
filename2 = "depth/RealSense_11.npy"
filename3 = "depth/RealSense_13.npy"
filename4 = "depth/RealSense_15.npy"
filename5 = "depth/RealSense_16.npy"

filename_empty = "depth/RealSense_try2_2.npy"


def point_finding(sample_img, plot_title="click to copy the point"):
    import pyperclip
    fig, ax = plt.subplots()
    plt.title(plot_title)
    ax.imshow(sample_img)

    def on_click(event):
        # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #       ('double' if event.dblclick else 'single', event.button,
        #        e  vent.x, event.y, event.xdata, event.ydata))
        print(f"[{int(round(event.xdata, 0))},{int(round(event.ydata, 0))}], {sample_img[int(round(event.ydata, 0))][int(round(event.xdata, 0))]}")
        pyperclip.copy(f"[{int(round(event.xdata, 0))},{int(round(event.ydata, 0))}]")

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()


def get_image(filename=filename1):
    return np.load(filename)


class Vehicle_detection_depth_cam():
    filename1 = "depth/RealSense_4.npy"
    filename2 = "depth/RealSense_11.npy"
    filename3 = "depth/RealSense_13.npy"
    filename4 = "depth/RealSense_15.npy"
    filename5 = "depth/RealSense_16.npy"

    filename_empty = "depth/RealSense_try2_2.npy"

    def __init__(self):
        pass

    def saliency(self):

        image = get_image(self.filename1)
        empty = get_image(self.filename_empty)
        # image = empty

        # initialize OpenCV's objectness saliency detector and set the path
        # to the input model files
        saliency = cv2.saliency.ObjectnessBING_create()
        saliency.setTrainingPath("model")

        # compute the bounding box predictions used to indicate saliency
        (success, saliencyMap) = saliency.computeSaliency(image)
        numDetections = saliencyMap.shape[0]

        # loop over the detections
        for i in range(0, min(numDetections, 10)):
            # extract the bounding box coordinates
            (startX, startY, endX, endY) = saliencyMap[i].flatten()

            # randomly generate a color for the object and draw it on the image
            output = image.copy()
            color = np.random.randint(0, 255, size=(3,))
            color = [int(c) for c in color]
            cv2.rectangle(output, (startX, startY), (endX, endY), color, 2)
            # show the output image
            cv2.imshow("Image", output)
            cv2.waitKey(0)

    def detect(self):
        # frame = cv2.imread('depth/RealSense2_178.jpg', CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(frame, alpha=0.1), cv2.COLORMAP_JET)
        image = get_image(self.filename1)
        empty = get_image(self.filename_empty)

        image[image > 1500] = 3000
        empty[empty > 1500] = 3000

        image1 = empty - image
        image1[image1 > 3000] = 0

        # image = image1


        ## threshholding
        a = (255 / image.max())
        print("a= ", a)
        image = cv2.convertScaleAbs(image, alpha=a)
        # point_finding(image)
        img_thrs = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY)
        cv2.imshow("thresh", img_thrs[1])
        cv2.waitKey()
        img_thrs2 = cv2.threshold(image, 126, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow("thresh", img_thrs2[1])
        cv2.waitKey()
        # img = img_thrs[1]*img_thrs2[1]
        img = cv2.bitwise_and(img_thrs[1],img_thrs2[1])
        img = cv2.bitwise_not(img)
        cv2.imshow("thresh", img)
        cv2.waitKey()

        image = img

        # a = (255 / image.max())
        # print("a= ", a)
        # image = cv2.convertScaleAbs(image, alpha=a)

        image2 = cv2.applyColorMap(image, cv2.COLORMAP_HSV)

        # image1 = cv2.convertScaleAbs(image, alpha=0.07)
        # image1 = cv2.applyColorMap(image1, cv2.COLORMAP_HSV)
        #
        # cv2.imshow("filtered", image1)
        # cv2.waitKey()

        # image = cv2.GaussianBlur(image, (5, 5), 2)

        # image2 = cv2.convertScaleAbs(image, alpha=0.07)
        # image2 = cv2.applyColorMap(image2, cv2.COLORMAP_HSV)

        cv2.imshow("blured", image2)
        cv2.waitKey()




        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 0
        params.maxThreshold = 255

        # Filter by Area.
        params.filterByArea = False
        params.minArea = 1000

        # Filter by Circularity
        params.filterByCircularity = False
        params.minCircularity = 0.2

        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = 0.87

        # Filter by Inertia
        params.filterByInertia = False
        params.minInertiaRatio = 0.1

        # by color
        params.filterByColor = True

        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3:
            detector = cv2.SimpleBlobDetector(params)
        else:
            detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(image)
        print("keypoins length= ", len(keypoints))
        im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Keypoints", im_with_keypoints)
        cv2.waitKey(0)


def canny(image):
    # Returns the processed image
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # cv2.imshow("gray", gray)
    # cv2.waitKey()
    image = cv2.GaussianBlur(image, (5, 5), 2)
    # cv2.imshow("blur", blur)
    # cv2.waitKey()
    image = cv2.Canny(image, 100, 200)
    # cv2.imshow("canny", canny)
    # cv2.waitKey()
    return image


# point_finding(get_image(filename1))
v = Vehicle_detection_depth_cam()
# v.saliency()
v.detect()
