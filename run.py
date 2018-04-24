from lib import Rovio
import cv2
import numpy as np
import sys, time
import os
from skimage import img_as_ubyte

from yolo.frondend import YOLO

# ------------------------------ Configuration Setting ------------------------------------------ #
config = {
    "model": {
        "architecture": "Full Yolo",
        "input_size": 416,
        "anchors": [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
        "max_box_per_image": 10,
        "labels": ["rovio"]
    }
}

labels = config['model']['labels']

rovio_detector_w_path = os.path.join('rovio-detector-weights', 'rovio_detector_weights.06-0.02.h5')


# ------------------------------------------------------------------------------------------------ #


class rovioControl(object):
    def __init__(self, url, username, password, config, port=80):
        # Initialize the robot with username,pw, ip
        self.rovio = Rovio(url, username=username, password=password,
                           port=port)
        self.last = None
        self.key = 0
        self.config = config
        self.rovio_detector = self.setup_rovio_detector()

    def night_vision(self, frame):
        # Night Vision is convert to grayscale and histogram equalization
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.equalizeHist(frame)
        return frame

    def show_battery(self, frame):
        # Get Battery Percentage into frame
        sh = frame.shape
        m, n = sh[0], sh[1]
        battery, charging = self.rovio.battery()
        battery = 100 * battery / 130.
        bs = "Battery: %0.1f" % battery
        cs = "Status: Roaming"
        if charging == 80:
            cs = "Status: Charging"
        cv2.putText(frame, bs, (20, 20),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))

        cv2.putText(frame, cs, (300, 20),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))

        return frame

    def resize(self, frame, size=(640, 480)):
        # Resize
        frame = cv2.resize(frame, size)
        return frame

    #############################################################################################
    #								IMPLEMENTATION OF OWN FUNCTIONS								#
    #############################################################################################
    # Face detection, Floor Finder	

    #########################################################################################
    #								FACE DETECTION algorithm								#
    #########################################################################################	
    def face_detection(self):
        # Raise Rovio Head to detect face
        self.rovio.head_up()
        # Default flag as false until detect a face
        flag = False
        while flag == False:
            # While cannot detect any face move forward and perform face detection
            self.rovio.step_forward()
            # Get frame from camera
            frame = self.rovio.camera.get_frame()
            ###########################################
            # 		Face Detection algorithm
            ###########################################
            # Cascade Classifier using haarcascade
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            # Turn the frame to gray scale first
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Perform multi scale Classification on the grayscaled image
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            # If face detected, draw rectangle on face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if faces == ():
                # If no face detected, skip
                pass
            else:
                # Else, show on screen face detected,
                # flag to true,
                # play sound and
                # write image to server
                cv2.imshow("Facedetector", frame)
                cv2.imwrite("face.png", frame)
                flag = True
                # winsound.PlaySound('%s.wav' % 'humandetected', winsound.SND_FILENAME)

    #########################################################################################
    #								Floor Finder algorithm									#
    #########################################################################################
    def floor_finder(self):
        # Capture image
        frame = self.rovio.camera.get_frame()
        # Perform Gaussian Blur to reduce noise
        gaussian = cv2.GaussianBlur(frame, (5, 5), 0)
        # Perform Canny Edge Detection to detect edges and lines
        edges = cv2.Canny(gaussian, 100, 200)
        ##############################
        # Make the line more obvious #
        ##############################
        # Kernel
        kernel = np.ones((2, 2), np.uint8)
        # Dilation
        dilate = cv2.dilate(edges, kernel, iterations=1)
        # Covert to array
        im = np.asarray(dilate)
        # Get Height
        h = np.size(im, 0)
        # Get Width
        w = np.size(im, 1)
        y = 0
        line = []
        #################################################
        #    If found edges, then draw rectangle		#
        #################################################
        # Please improve this: Still can improve		
        for j in range(h - 1, 0, -1):
            for i in range(w):
                if not im[j][i] == 0:
                    y = j
                    break
            cv2.rectangle(frame, (0, y), (w, h), (245, 252, 0), 2)
        cv2.imshow("Zone", frame)
        # Return the height of the zone
        return h - y

    def setup_rovio_detector(self):
        yolo = YOLO(architecture=config['model']['architecture'],
                    input_size=config['model']['input_size'],
                    labels=config['model']['labels'],
                    max_box_per_image=config['model']['max_box_per_image'],
                    anchors=config['model']['anchors'])

        print('Loading model {}'.format(rovio_detector_w_path))
        yolo.load_weights(rovio_detector_w_path)

        return yolo

    # MAIN FUNCTION TO BE CALLED
    def main(self):
        # Whenever initialize, raise head to middle
        # self.rovio.head_middle()

        # Get frame and show original capture frame
        frame = self.rovio.camera.get_frame()
        cv2.imshow("Original", frame)

        if not isinstance(frame, np.ndarray):
            return

        ori_frame = frame
        frame = self.night_vision(frame)
        frame = self.resize(frame)
        ori_frame = self.resize(ori_frame)
        frame = cv2.merge([frame, frame, frame])
        frame = self.show_battery(frame)


        # ROVIO detect start here
        # keep rotate right to search for Rovio
        boxes = self.rovio_detector.predict(ori_frame)
        if len(boxes) < 1:
            self.rovio.rotate_right(angle=15, speed=1)
        else:
            # Get the nearest one to move to (Biggest Area)
            x, y, w, h = 0, 0, 0, 0
            max_box_i = 0
            max_area = 0
            for index, box in enumerate(boxes):
                width = box.w + box.x
                height = box.h + box.y

                area = (box.w + box.x) * (box.h + box.y)
                print(width / height)
                if max_area < area and (width/height > 1.1 and width/height < 1.2):
                    max_area = area
                    max_box_i = index

            x, y, w, h = boxes[max_box_i].get_position()

            # get center point of the box
            xmin = int((box.x - box.w / 2) * frame.shape[1])
            xmax = int((box.x + box.w / 2) * frame.shape[1])
            ymin = int((box.y - box.h / 2) * frame.shape[0])
            ymax = int((box.y + box.h / 2) * frame.shape[0])

            cv2.rectangle(ori_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            cv2.putText(ori_frame,
                        labels[box.get_label()] + ' ' + str(box.get_score()),
                        (xmin, ymin - 13),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1e-3 * frame.shape[0],
                        (0, 255, 0), 2)

            cv2.imshow('detector', ori_frame)

            # Assume x and y is the center point
            if(x * frame.shape[0] >= 213 and x * frame.shape[0]<= 426):
                self.rovio.forward()
            elif x * frame.shape[1]> frame.shape[1] / 2:
                self.rovio.rotate_right(angle=15, speed=1)
            else:
                self.rovio.rotate_left(angle=15, speed=1)

                #####################################################
                #				Perform Floor floor_finder			#
                #####################################################
                # If safe zone is more than 80 then check for infrared detection

                if self.floor_finder() > 80:
                    pass
                    # if(not self.rovio.ir()):
                    #     self.rovio.api.set_ir(1)
                    # if (not self.rovio.obstacle()):
                    #     self.rovio.forward()
                    #     self.rovio.forward()
                    #     self.rovio.forward()
                    #     self.rovio.forward()
                    #     self.rovio.forward()
                    #     self.rovio.forward()
                    # else:
                    #     self.rovio.rotate_right(angle=20, speed=2)
                    # Rotate right is safe zone is smaller than 80 pixels
                else:
                    pass
                    # self.rovio.rotate_right(angle=20, speed=2)

                # If Button Pressed, onAction
                # Use ASCII for decode
                self.key = cv2.waitKey(20)
                if self.key > 0:
                # print self.key
                    pass
                if self.key == 114:  # r
                    self.rovio.turn_around()
                elif self.key == 63233 or self.key == 115:  # down or s
                    self.rovio.backward(speed=7)
                elif self.key == 63232 or self.key == 119:  # up or w
                    self.rovio.forward(speed=1)
                elif self.key == 63234 or self.key == 113:  # left or a
                    self.rovio.rotate_left(angle=12, speed=5)
                elif self.key == 63235 or self.key == 101:  # right or d
                    self.rovio.rotate_right(angle=12, speed=5)
                elif self.key == 97:  # left or a
                    self.rovio.left(speed=1)
                elif self.key == 100:  # right or d
                    self.rovio.right(speed=1)
                elif self.key == 44:  # comma
                    self.rovio.head_down()
                elif self.key == 46:  # period
                    self.rovio.head_middle()
                elif self.key == 47:  # slash
                    self.rovio.head_up()
                elif self.key == 32:  # Space Bar, pressed then perform face detection
                    flag = False
                # self.rovio.stop()
                # self.face_detection()

                #############################################################################
                #						Main Class and rovioControl							#
                #############################################################################
                # TODO: Edit the IP address here
if __name__ == "__main__":
    url = '192.168.43.2'
    user = 'rovio'
    password = "azwan9669"
    app = rovioControl(url, user, password, config)




while True:
    app.main()
    # If press esc, then head down and stop the loop
    if app.key == 27:
        app.rovio.head_down()
        break
