# from PIL import Image
import cv2
import numpy as np


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

        image = self.get_image(self.filename3)
        empty = self.get_image(self.filename_empty)
        image = empty

        # initialize OpenCV's objectness saliency detector and set the path
        # to the input model files
        saliency = cv2.saliency.ObjectnessBING_create()
        saliency.setTrainingPath("model")
        
        # compute the bounding box predictions used to indicate saliency
        (success, saliencyMap) = saliency.computeSaliency(image)
        numDetections = saliencyMap.shape[0]

        # loop over the detections
        for i in range(0, min(numDetections,10)):
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
        image = self.get_image(self.filename3)
        empty = self.get_image(self.filename_empty)

        image[image > 1500] = 3000
        empty[empty > 1500] = 3000

        image1 = empty - image
        image1[image1 > 3000] = 0



        image = image1


        a = (255 / image.max())
        print("a= ",a )
        image = cv2.convertScaleAbs(image, alpha=a)

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
        params.minThreshold = 10
        params.maxThreshold = 200

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 1000

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.87

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.01

        # by color
        # params.filterByColor = True

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


    def get_image(self, filename=filename1):
            return np.load(filename)


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


v = Vehicle_detection_depth_cam()
v.saliency()
# v.detect()
