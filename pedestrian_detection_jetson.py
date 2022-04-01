import jetson.inference
import jetson.utils

import argparse
import sys

class PedestrianDetectionJetson:

    # parse the command line
    # parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
    #                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.poseNet.Usage() +
    #                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

    # parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
    # parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
    # parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
    # parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
    # parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 

    # opt = parser.parse_known_args()[0]

    def __init__(self):

        # load the pose estimation model
        self.net = jetson.inference.poseNet("resnet18-body", sys.argv, 0.15)

        # create video sources & outputs
        # self.output = jetson.utils.videoOutput('my_video.mp4') # 'my_video.mp4' for file

        self.input = jetson.utils.videoSource('/home/nvidia/Desktop/VROOM_Sandbox/pose.png')
        

    def detectPedestrian(self):

        try: 
            # perform pose estimation (with overlay)
            img = self.input.Capture()
            print(img.type)
            print('Before posing')
            poses = self.net.Process(img)#, overlay="links,keypoints")
            print('After posing')

            # print the pose results
            print("detected {:d} objects in image".format(len(poses)))

            for pose in poses:
                print(pose)
                print(pose.Keypoints)
                print('Links', pose.Links)

            # render the image
            # self.output.Render(img)

            # # update the title bar
            # self.output.SetStatus("{:s} | Network {:.0f} FPS".format("resnet18-body", self.net.GetNetworkFPS()))
            print('FPS: ', self.net.GetNetworkFPS())

            # print out performance info
            self.net.PrintProfilerTimes()

            return img
        except Exception as e:
            print(e)