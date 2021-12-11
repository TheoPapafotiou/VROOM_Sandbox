import numpy as np
import cv2
import math
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl

mpl.use('macosx')

# helps finding the coordinates
import pyperclip

"""
This class represents a mask for the images, and therefore the FOV of the car.
"""


class Mask:
    mask = {"circles": [],
            "polygons": []}

    # polmask1 = [[214, 164], [463, 178], [615, 436], [44, 444]]
    # polymask2 = [[50, 50], [50, 150], [150, 150], [150, 50]]
    # The points that represent the mask's shape, in the form of np array [x0,y0,x1,y1,x2,y2...]
    polygons = {}

    # circle points
    # the data will be (point_x, point_y, radius)
    # circle1 = [100, 100, 20]
    circles = {}

    # The stencil used for masking. A matrix of shape WIDTH x SHAPE x 1 (as it has no channels)
    stencil = []

    # Constructor of the Mask.
    # @args:
    # sample_img: the sample image to determine the stencil
    # num_of_points: The number of points the mask will have.
    # input_img_sample: A sample of the input images, so the dimensions of the stencil can be determined.

    def __init__(self, sample_img, filename, circle_data=None, polygons_data=None):
        self.filename = filename
        self.shape = sample_img.shape[0:2]
        self.stencil = np.zeros(self.shape, dtype=np.uint8)
        if filename != "":
            import os.path
            if os.path.isfile(filename):
                with open(filename, "r") as f:
                    self.mask = json.load(f)
                self.update_stencil()
            else:
                with open(filename, "w") as f:
                    json.dump(self.mask, f, indent=4)
                self.easy_setup(sample_img=sample_img)
        else:
            if circle_data is not None:
                self.make_circle_mask(circle_data)
            if polygons_data is not None:
                self.make_polygon_mask(polygons_data)
        # self.points = np.zeros(num_of_points * 2)

    def update_file(self):
        with open(self.filename, "w") as f:
            json.dump(self.mask, f, indent=4)

    def point_finding(self, sample_img, plot_title="click to copy the point", with_mask=True):
        fig, ax = plt.subplots()
        plt.title(plot_title)
        if with_mask:
            ax.imshow(self.apply_mask(sample_img))
        else:
            ax.imshow(sample_img)

        def on_click(event):
            # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
            #       ('double' if event.dblclick else 'single', event.button,
            #        e  vent.x, event.y, event.xdata, event.ydata))
            pyperclip.copy(f"[{int(round(event.xdata, 0))},{int(round(event.ydata, 0))}]")

        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()

    def add_circle(self, x_y_r=None, ui=True, sample_img=None):
        if ui:
            self.point_finding(sample_img, "click to set the center and close", with_mask=False)
            points = input("copy the point !SEPERATED BY SPACE!")
            points = points.split()
            radius = input("give the radius in pixels:")
            self.mask["circles"].append([int(points[0]), int(points[1]), int(radius)])
            self.update_file()

        if x_y_r is None:
            pass
        else:
            self.mask["circles"].append(x_y_r)
            self.update_file()

    def add_polygon_ui(self, sample_image=None):
        from matplotlib.widgets import Button
        fig, ax = plt.subplots()
        plt.imshow(sample_image, zorder=0)
        stencil_cp = self.stencil.copy()
        polygons_cp = self.mask["polygons"].copy()
        # plt.imshow(self.apply_mask(sample_image), zorder=0)
        temp_poly = []
        while plt.fignum_exists(1):
            temp_points = []
            print("pass while")
            axpB = plt.axes([0.5, 0, 0.15, 0.075])
            axcB = plt.axes([0.75, 0, 0.15, 0.075])
            bpoly = Button(axpB, 'SAVE POLY')

            # exitt = False
            def savee(event):
                # self.mask["polygons"] = polygons_cp
                print(f"temp poly: {temp_poly}")
                # temp_poly = temp_poly.copy(.pop()
                print(f"temp poly2: {temp_poly[:-1]}")
                self.mask["polygons"].append(temp_poly[:-1])
                # print(temp_poly)
                self.update_file()
                print("CLOSED00")
                plt.clf()
                print("CLOSED11")
                plt.close(fig)
                print("CLOSED")
                # exitt = True

            bpoly.on_clicked(savee)
            bcirc = Button(axcB, 'WITHDRAW')

            def wd(event):
                self.mask["polygons"] = polygons_cp
                self.stencil = stencil_cp
                plt.clf()
                plt.close(fig)
                print("CLOSED")
                # exitt = True

            bcirc.on_clicked(wd)
            print("pass")

            def on_click(event):
                # temp_points.append([int(round(event.xdata, 0)), int(round(event.ydata, 0))])
                temp_points = ([int(round(event.xdata, 0)), int(round(event.ydata, 0))])
                # temp_stencil = self.stencil.copy()
                # self.add_polygon(points=temp_points)
                # ax.imshow(self.apply_test_mask(sample_image, temp_stencil))
                # temp_poly = polygons_cp
                temp_poly.append(temp_points)
                print(f"added {temp_points} ")
                ax.plot(temp_points[0], temp_points[1], 'ro', zorder=5)
                # for p in temp_poly:
                #     # plt.plot(p[0], p[1])
                #     print(f"added {temp_points} ")
                #     ax.plot(p[0], p[1], 'ro', zorder=5)
                #     # np.append(temp_stencil,temp_points)

            fig.canvas.mpl_connect('button_press_event', on_click)
            plt.waitforbuttonpress()
            # ax.imshow(self.apply_mask(sample_image), zorder=0)
            plt.show()

    # Draw a circle on the stencil
    # @args:
    # circle: data that looks like this (point_x, point_y, radius)
    def make_circle_mask(self, circles):
        if circles is None:
            return 0
        for c in circles:
            c = np.array(c)
            cv2.circle(self.stencil, c[0:2], c[2], (255, 255, 255), -1)
            # cv2.imshow("rrr", self.stencil)
            # cv2.waitKey(0)

    def make_polygon_mask(self, polygons):
        if polygons is None:
            return 0
        # npolys = np.array(len(polygons))
        for p in polygons:
            poly = np.array(p)
            cv2.fillConvexPoly(self.stencil, poly, color=(255, 255, 255))
        # cv2.polylines(self.stencil, polygon,1, color =(0,0,0))
        # cv2.imshow("rrr", self.stencil)
        # cv2.waitKey(0)
        # o_i = cv2.bitwise_and(input_image[:, :], input_image[:, :], mask=self.stencil)
        # cv2.imwrite("test.jpg", o_i)  # save frame as JPEG file
        # return o_i

    def update_stencil(self):
        self.make_circle_mask(circles=self.mask["circles"])
        self.make_polygon_mask(polygons=self.mask["polygons"])

    #
    # def apply_test_mask(self, input_image, test_mask):
    #     masked = cv2.bitwise_and(input_image,
    #                              input_image,
    #                              mask=test_mask)
    #     return masked

    def apply_mask(self, input_image):
        masked = cv2.bitwise_and(input_image,
                                 input_image,
                                 mask=self.stencil)
        return masked

    def easy_setup(self, sample_img):
        while True:
            in1 = input("----------------------------\n"
                        "press P to add a polygon\n"
                        "press C to add a circle\n"
                        "press X to end the setup\n")
            if in1.lower() == 'x':
                break
            elif in1.lower() == 'p':
                self.add_polygon_ui(sample_image=sample_img)
            elif in1.lower() == 'c':
                self.add_circle(sample_img=sample_img)
            else:
                print("try again")




# testtt

image = cv2.imread("testdiko.jpg")
dims = image.shape
image_np = np.copy(image)
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
m = Mask(image, "mask_test29")
# m.point_finding(image)
# m.add_polygon_mask(m.polymask1)
# m.add_circle_mask(m.circle1)
# m.set_polygon(150, 200, 5)
# m.add_polygon_ui(sample_image=image)
# m.add_circle(sample_img=image)
# m.easy_setup(sample_img=image)
masked = m.apply_mask(image)
# masked = m.apply_to_img(gray_img)
# cv2.imshow("test", image)
# cv2.waitKey()
cv2.imshow("test", masked)
cv2.waitKey()
# plt.show()
# print("test")
