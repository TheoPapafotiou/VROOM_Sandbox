import unittest
import cv2

from ...src.lane_keeping import LaneKeeping

class TestLaneKeeping(unittest.TestCase):
    def setUp(self):
        self.laneKeeping = LaneKeeping()
        self.desired_angles = [0.22, 0.03, 0.06, -0.01]
        self.path = "Lane_keeping_inputs/image_" 

    def TestLaneKeeping(self):
        try:
            for i in range(0, 4):
                image = cv2.imread(self.path+str(i))
                angle = self.laneKeeping(image)
                message = "Lane keeping failed in image_" + str(i)
                self.assertAlmostEqual(angle/100, self.desired_angles[i], 1, msg=message)
        except Exception as e:
            print(e)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
