import cv2
import numpy as np
import time

class SignDetection:
    
    """
    This class implements the sign & traffic lights detection procedure in our vehicle.
    For the detection a trained tiny-YOLO deep neural network is used.
    """

    def __init__(self):

        self.classes = ['ParkingSpot','Crosswalk','Ahead','HighwayEnd','HighwayStart','PriorityRoad','Stop','NoEntry','Roundabout','TrafficLights']
        self.net = cv2.dnn.readNetFromDarknet("yolov3_tiny-custom.cfg",r"yolov3_tiny-custom_total.weights")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # Initial Set-up
        init_img = cv2.imread("Init_img.png")
        blob = cv2.dnn.blobFromImage(init_img, 1/255,(416,416),(0,0,0),swapRB = True,crop= False)
        self.net.setInput(blob)
        output_layers_name = self.net.getUnconnectedOutLayersNames()
        layerOutputs = self.net.forward(output_layers_name)

        self.label = "Something"
        self.distance = 0
        self.x_camera = 640
        self.y_camera = 480
        self.center_x = 0
        self.center_y = 0
        self.confidence_limit = 0.9
        self.nms_param1 = .8
        self.nms_param2 = .4

    def detectSignProcedure(self, blob, img, height, width):
        self.net.setInput(blob)
        output_layers_name = self.net.getUnconnectedOutLayersNames()
        layerOutputs = self.net.forward(output_layers_name)
        self.label = 'Something'
        self.distance = 0
        self.confidence = 0

        boxes =[]
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                score = detection[5:]
                class_id = np.argmax(score)
                confidence = score[class_id]
                if confidence >= self.confidence_limit:
                    self.center_x = int(detection[0] * width)
                    self.center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3]* height)
                    x = int(self.center_x - w/2)
                    y = int(self.center_y - h/2)
                    boxes.append([x,y,w,h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes,confidences,self.nms_param1,self.nms_param2)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0,255,size =(len(boxes),3))
        if  len(indexes)>0:
            for i in indexes.flatten():
                x,y,w,h = boxes[i]
                self.label = str(self.classes[class_ids[i]])
                self.confidence = str(round(confidences[i],2))
                self.distance = self.y_camera - self.center_y
                #print("I found a " + self.label + " sign with confidence " + confidence + " at a distance: " + str(self.distance))
                color = colors[i]
                cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                cv2.putText(img,self.label, (x-int(self.x_camera/2),y+int(self.y_camera/4)),font,2,color,2)
                cv2.putText(img,"Confidence: " + self.confidence, (x-int(self.x_camera/2),y+int(self.y_camera/3)),font,2,color,2)

        return self.label, self.distance

    def detectSign(self, img, height, width):
        blob = cv2.dnn.blobFromImage(img, 1/255,(416,416),(0,0,0),swapRB = True,crop= False)
        label, distance = self.detectSignProcedure(
            blob,
            img,
            height,
            width
        )

        result = {'Label': str(label), 'Distance': str(distance)}
        # result = json.dumps(result)

        return result