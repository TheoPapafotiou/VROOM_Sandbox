from PIL import Image
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

    def detect(self):
        # frame = cv2.imread('depth/RealSense2_178.jpg', CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(frame, alpha=0.1), cv2.COLORMAP_JET)
        image = self.get_image(self.filename4)
        empty = self.get_image(self.filename_empty)

        im_minus_empt = image - empty

        emp_minus_im =  empty - image
        im_emp = zip(image, empty)

        # diff_array = [x for x, y in im_emp if abs(x-y) > 1000 ]
        # diff_array = cv2.applyColorMap(cv2.convertScaleAbs(diff_array, alpha=0.07), cv2.COLORMAP_HSV)

        # cv2.imshow("1", diff_array)
        # cv2.waitKey()

        # im_minus_empt = cv2.convertScaleAbs(im_minus_empt, alpha=0.5)
        # emp_minus_im = cv2.convertScaleAbs(emp_minus_im, alpha=0.5)
        # im_minus_empt = cv2.convertScaleAbs(im_minus_empt, alpha=0.5)
        # emp_minus_im = cv2.convertScaleAbs(emp_minus_im, alpha=0.5)

        # for a in [0.01, 0.05, 0.07, 0.1, 0.2, 0.4, 0.5, 0.75, 1, 1.2, 1.5, 2]:
        #     emp_minus_im_clrmp =  cv2.applyColorMap(cv2.convertScaleAbs(emp_minus_im, alpha=a), cv2.COLORMAP_HSV)
        #     # im_minus_empt_colormap = cv2.applyColorMap(cv2.convertScaleAbs(im_minus_empt, alpha=a), cv2.COLORMAP_HSV)
        #
        #     depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(image, alpha=0.1), cv2.COLORMAP_JET)
        #
        #
        #     cv2.imshow("1", emp_minus_im_clrmp)
        #     cv2.waitKey()
        #     #
        #     # cv2.imshow("2", im_minus_empt_colormap)
        #     # cv2.waitKey()


        image2 = []
        # detector = cv2.SimpleBlobDetector()

        # keypoints = detector.detect(depth_colormap)
        # detector.detect(empty)
        # im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),
        #                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        # params.minThreshold = 10
        # params.maxThreshold = 200

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 2000

        # Filter by Circularity
        params.filterByCircularity = False
        params.minCircularity = 0.1

        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = 0.87

        # Filter by Inertia
        params.filterByInertia = False
        params.minInertiaRatio = 0.01

        # by color
        # params.filterByColor = True

        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3:
            detector = cv2.SimpleBlobDetector(params)
        else:
            detector = cv2.SimpleBlobDetector_create(params)

        for i in ["try2_2", 4, 11, 13, 15, 16]:
            empty = self.get_image(self.filename_empty)
            image = np.load(f"depth/RealSense_{i}.npy")
            cv2.imshow("raw", image)
            cv2.waitKey()
            emp_minus_im = empty - image
            
            image = cv2.convertScaleAbs(emp_minus_im, alpha=0.07)
            cv2.imshow("scale abs", image)
            cv2.waitKey()

            # image = cv2.convertScaleAbs(image, alpha=0.07)
            # cv2.imshow("scale abs", image)
            # cv2.waitKey()

            image = cv2.applyColorMap(image, cv2.COLORMAP_HSV)
            cv2.imshow("colormap", image)
            cv2.waitKey()



            # for a in np.arange(0.05, 2, 0.05):
            #     print("a= ", a)
            #     empty = self.get_image(self.filename_empty)
            #     image = np.load(f"depth/RealSense_{i}.npy")
            #     im_minus_empt = image - empty
            #     emp_minus_im = empty - image
            #     # cv2.imshow("1", emp_minus_im)
            #     # cv2.waitKey()
            #
            #     im_minus_empt = cv2.convertScaleAbs(im_minus_empt, alpha=a)
            #     emp_minus_im = cv2.convertScaleAbs(emp_minus_im, alpha=a)
            #
            #     im_minus_empt = cv2.convertScaleAbs(im_minus_empt, alpha=a)
            #     emp_minus_im = cv2.convertScaleAbs(emp_minus_im, alpha=a)
            #
            #     emp_minus_im_clrmp = cv2.applyColorMap(emp_minus_im, cv2.COLORMAP_JET)
            #     im_minus_empt_colormap = cv2.applyColorMap(im_minus_empt, cv2.COLORMAP_JET)
            #
            #     cv2.imshow("1", emp_minus_im_clrmp)
            #     cv2.waitKey()
            #
            #     cv2.imshow("2", im_minus_empt_colormap)
            #     cv2.waitKey()



            # AUTA TA DYO MAZI KATI MPOROUN NA KANOUN GIA NA PROSTHETOUN INFO
            # ## kanei kati periergo gia to blur kai xanei info
            # image = cv2.GaussianBlur(image, (5, 5), 2)
            # cv2.imshow("blur", image)
            # cv2.waitKey()
            #
            # ## canny thn canny(lol) aspromavrh kai den mas kanny katholou
            # ## mporei kapws na xrhsimopoihthei gia kathodighsh tho
            # image = cv2.Canny(image, 100, 200)
            # cv2.imshow("canny", image)
            # cv2.waitKey()

            ## kanei kati periergo gia to blur kai xanei info



            # image = cv2.GaussianBlur(image, (5, 5),2)
            # cv2.imshow("blur", image)
            # cv2.waitKey()

            # ## canny thn canny(lol) aspromavrh kai den mas kanny katholou
            # ## mporei kapws na xrhsimopoihthei gia kathodighsh tho
            # canny = cv2.Canny(image, 100, 200)
            # cv2.imshow("canny", image)
            # cv2.waitKey()

            # wide = cv2.Canny(image, 10, 200)
            # mid = cv2.Canny(image, 30, 150)
            # tight = cv2.Canny(image, 240, 250)
            # # show the output Canny edge maps
            # cv2.imshow("Wide Edge Map", wide)
            # cv2.imshow("Mid Edge Map", mid)
            # cv2.imshow("Tight Edge Map", tight)
            # cv2.waitKey(0)

            # threshold
            # thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)[1]
            #
            # # get the (largest) contour
            # contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # contours = contours[0] if len(contours) == 2 else contours[1]
            # big_contour = max(contours, key=cv2.contourArea)
            #
            # # draw white filled contour on black background
            # result = np.zeros_like(image)
            # cv2.drawContours(result, [big_contour], 0, (255, 255, 255), cv2.FILLED)
            #
            # image = np.concatenate((canny, image), axis=0)
            #
            # cv2.imshow("contour", image)
            # cv2.waitKey()

            # image = cv2.applyColorMap(image, cv2.COLORMAP_HSV)
            # cv2.imshow("col map", image)
            # cv2.waitKey()
            # colormap = image


            keypoints = detector.detect(image)
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
v.detect()
