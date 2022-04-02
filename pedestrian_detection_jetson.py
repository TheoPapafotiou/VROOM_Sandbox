import jetson.inference
import jetson.utils

import cv2
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

    def __init__(self):

        # load the pose estimation model
        self.net = jetson.inference.poseNet("resnet18-body", argv=["--log-level=error"], threshold=0.15)        

    def detectPedestrian(self, numpy_img):

        try: 
            # perform pose estimation (with overlay)
            cuda_img = jetson.utils.cudaFromNumpy(numpy_img)
            print(type(numpy_img), 'VS', type(cuda_img))
            poses = self.net.Process(cuda_img, overlay="links,keypoints")

            # print the pose results
            print("detected {:d} objects in image".format(len(poses)))

            for pose in poses:
                print(pose)
                print(pose.Keypoints)
                print('Links', pose.Links)

            return_img = jetson.utils.cudaToNumpy(cuda_img)
            print('FPS: ', self.net.GetNetworkFPS())
            
            links = [len(pose.Links) for pose in poses] 
            print(poses[0])

            if len(links) >= 1: 

                final_pose_index = links.index(max(links))
                final_pose = poses[final_pose_index]

            # print out performance info
            # self.net.PrintProfilerTimes()

            return return_img

        except Exception as e:
            print(e)