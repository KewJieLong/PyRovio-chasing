from lib import Rovio
import cv2
import numpy as np
import sys, time
import os
from skimage import img_as_ubyte
import threading
from node import Node

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

rovio_detector_w_path = os.path.join('rovio-detector-weights', 'rovio_detector_weights.01-0.03.h5')
winning_threshold = 0.16
patient_steps_no_forward = 3
steps_reset_safe_zone = 4
num_safe_zone = 8

# ------------------------------------------------------------------------------------------------ #


rovio = {
    'rovio1_ready': False,
    'rovio2_ready': False,
}




class rovioControl(object):
    def __init__(self, name, url, username, password, rovio_detector, rovio_ready, chaser, port=80):
        # Initialize the robot with username,pw, ip
        self.rovio = Rovio(url, username=username, password=password,
                           port=port)
        self.last = None
        self.key = 0
        self.rovio_detector = rovio_detector
        self.chaser = chaser
        self.rovio_ready = rovio_ready
        self.name = name
        self.steps_no_forward = 0
        self.head = None
        self.init_safe_zone_ring()
        self.steps_moving = 0

    def init_safe_zone_ring(self):
        self.head = Node(True)
        current = self.head
        for i in range(num_safe_zone - 1):
            new_node = Node(True)
            current.tail = new_node
            current = new_node

        current.tail = self.head

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
        # cv2.imshow("Zone", frame)
        # Return the height of the zone
        return h - y

    def toggle_chaser(self):
        self.chaser = not self.chaser

    def start(self):
        self.search_rovio()
        while rovio['rovio1_ready']:
            print('both rovio is ready!!!!!!')

            for i in range(3):
                print(i)
                time.sleep(1)

            while self.main():
                pass

            break

    def search_safe_zone(self):
        current = self.head
        count = 0
        while current.element is not True:
            print('Searching for safe zone ~~~ ')
            self.rovio.rotate_right(angle=45, speed=1)
            self.rovio.rotate_right(angle=45, speed=1)
            self.rovio.rotate_right(angle=45, speed=1)
            self.rovio.rotate_right(angle=45, speed=1)
            current = current.tail
            count += 1

            if count > num_safe_zone:
                print('HELP !!!!! i am trapped')
                break

        self.head = current
        return True



    # MAIN FUNCTION TO BE CALLED
    def main(self):
        print(self.name + ' action ' + str(self.chaser))
        # Whenever initialize, raise head to middle
        # self.rovio.head_middle()

        # Get frame and show original capture frame
        frame = self.rovio.camera.get_frame()
        # cv2.imshow("Original", frame)

        if not isinstance(frame, np.ndarray):
            return

        ori_frame = frame
        frame = self.night_vision(frame)
        frame = self.resize(frame)
        ori_frame = self.resize(ori_frame)
        frame = cv2.merge([frame, frame, frame])
        frame = self.show_battery(frame)


        # ROVIO detect start here
        # keep rotate right to search for Rovio'


        # Assume x and y is the center point

        box = self.detect_rovio(ori_frame)
        if self.chaser:
            if box is not None:
                x, y, w, h = box.get_position()
                print('area for {}'.format(w * h))

                if(x * frame.shape[0] >= 160 and x * frame.shape[0] <= 480):
                    self.move()
                    if (w * h > winning_threshold):
                        print('Chaser win')
                        key = '{}_ready'.format(self.name)
                        self.rovio_ready[key] = False
                        return False

                elif x * frame.shape[1]> frame.shape[1] / 2:
                    self.rovio.rotate_right(angle=15, speed=1)
                    self.steps_no_forward += 1
                else:
                    self.rovio.rotate_left(angle=15, speed=1)
                    self.steps_no_forward += 1

                if self.steps_no_forward > patient_steps_no_forward:
                    self.rovio.forward()
                    self.rovio.forward()
                    self.rovio.forward()
                    self.rovio.forward()
                    self.steps_no_forward = 0

            else:
                if self.steps_no_forward < 10:
                    self.rovio.rotate_right(angle=15, speed=1)
                else:
                    self.rovio.forward()
                    self.rovio.forward()
                    self.rovio.forward()
                    self.steps_no_forward = 0

        if not self.chaser:
            if box is None:
                print('RUNNNNN')
                self.run(rotate_180=False)
            else:
                print('SAW ROVIO, turn 180 and RUNNN')
                self.head.element = False
                self.run(rotate_180=True)





            #############################################################################
            #						Main Class and rovioControl							#
            #############################################################################
            # TODO: Edit the IP address here

        return True

    def detect_rovio(self, frame, search_rovio=False):
        boxes = self.rovio_detector.predict(frame)
        if len(boxes) < 1:
            if search_rovio or self.chaser:
                self.rovio.rotate_right(angle=15, speed=1)
            return None
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
                print('y = {}'.format(box.y))
                if max_area < area and (width / height > 1.1 and width / height < 1.2) and (box.y > 0.5 and box.y < 0.7) :
                    max_area = area
                    max_box_i = index

            x, y, w, h = boxes[max_box_i].get_position()

            # get center point of the box
            xmin = int((box.x - box.w / 2) * frame.shape[1])
            xmax = int((box.x + box.w / 2) * frame.shape[1])
            ymin = int((box.y - box.h / 2) * frame.shape[0])
            ymax = int((box.y + box.h / 2) * frame.shape[0])

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            cv2.putText(frame,
                        labels[box.get_label()] + ' ' + str(box.get_score()),
                        (xmin, ymin - 13),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1e-3 * frame.shape[0],
                        (0, 255, 0), 2)

            cv2.imshow('{} detector'.format(self.name), frame)

            return boxes[max_box_i]

    def search_rovio(self):
        key = '{}_ready'.format(self.name)
        while not self.rovio_ready[key]:
            frame = self.rovio.camera.get_frame()
            box = self.detect_rovio(frame, search_rovio=True)
            if box is not None :
                x, y, w, h = box.get_position()
                if (x * frame.shape[0] >= 160 and x * frame.shape[0] <= 480):
                    print('found rovio in center')
                    key = '{}_ready'.format(self.name)
                    self.rovio_ready[key] = True
                    self.toggle_chaser()
                    break
                elif x * frame.shape[1] > frame.shape[1] / 2:
                    self.rovio.rotate_right(angle=15, speed=1)
                else:
                    self.rovio.rotate_left(angle=15, speed=1)

    def is_rovio_ready(self):
        key = '{}_ready'.format(self.name)
        return self.rovio_ready[key]

    def move(self):
        #####################################################
        #				Perform Floor floor_finder			#
        #####################################################
        # If safe zone is more than 80 then check for infrared detection

        while self.search_safe_zone():
            if not self.rovio.ir():
                self.rovio.api.set_ir(1)
            if not self.rovio.obstacle():
                self.steps_no_forward = 0
                self.rovio.forward()
                self.rovio.forward()
                self.rovio.forward()
                self.steps_moving += 1

                if self.steps_moving > steps_reset_safe_zone:
                    self.init_safe_zone_ring()
                    self.steps_moving = 0
                break
            else:
                print('detect Obstracle')
                self.head.element = False

        # if self.floor_finder() > 80:
        #     if(not self.rovio.ir()):
        #         self.rovio.api.set_ir(1)
        #     if (not self.rovio.obstacle()):
        #         self.rovio.forward()
        #         self.rovio.forward()
        #         self.rovio.forward()
        #         # self.rovio.forward()
        #         # self.rovio.forward()
        #         # self.rovio.forward()
        #     else:
        #         self.rovio.rotate_right(angle=20, speed=2)
        #     # Rotate right is safe zone is smaller than 80 pixels
        # else:
        #     self.rovio.rotate_right(angle=20, speed=2)

        # If Button Pressed, onAction
        # Use ASCII for decode
        # self.key = cv2.waitKey(20)
        # if self.key > 0:
        #     # print self.key
        #     pass
        # if self.key == 114:  # r
        #     self.rovio.turn_around()
        # elif self.key == 63233 or self.key == 115:  # down or s
        #     self.rovio.backward(speed=7)
        # elif self.key == 63232 or self.key == 119:  # up or w
        #     self.rovio.forward(speed=1)
        # elif self.key == 63234 or self.key == 113:  # left or a
        #     self.rovio.rotate_left(angle=12, speed=5)
        # elif self.key == 63235 or self.key == 101:  # right or d
        #     self.rovio.rotate_right(angle=12, speed=5)
        # elif self.key == 97:  # left or a
        #     self.rovio.left(speed=1)
        # elif self.key == 44:  # comma
        #     self.rovio.head_down()
        # elif self.key == 46:  # period
        #     self.rovio.head_middle()
        # elif self.key == 47:  # slash
        #     self.rovio.head_up()
        # elif self.key == 32:  # Space Bar, pressed then perform face detection
        #     flag = False
        # self.rovio.stop()
        # self.face_detection()

    def run(self, rotate_180=False):
        if rotate_180:
            self.turn_180()

        # self.rovio.forward()
        self.move()

    def turn_180(self):
        self.rovio.rotate_right(angle=20, speed=2)
        self.rovio.rotate_right(angle=20, speed=2)
        self.rovio.rotate_right(angle=20, speed=2)
        self.rovio.rotate_right(angle=20, speed=2)



    def reverse_backward(self):
        self.turn_180()
        for i in range(10):
            self.move()

    def backward(self):
        self.rovio.backward(speed=2)
        self.rovio.backward(speed=2)
        self.rovio.backward(speed=2)
        self.rovio.backward(speed=2)
        self.rovio.backward(speed=2)
        self.rovio.backward(speed=2)
        self.rovio.backward(speed=2)
        self.rovio.backward(speed=2)
        self.rovio.backward(speed=2)
        self.rovio.backward(speed=2)
        self.rovio.backward(speed=2)



def setup_rovio_detector(config):
    yolo = YOLO(architecture=config['model']['architecture'],
                input_size=config['model']['input_size'],
                labels=config['model']['labels'],
                max_box_per_image=config['model']['max_box_per_image'],
                anchors=config['model']['anchors'])

    print('Loading model {}'.format(rovio_detector_w_path))
    yolo.load_weights(rovio_detector_w_path)

    return yolo


if __name__ == "__main__":
    rovio_detector = setup_rovio_detector(config)


    url1 = '192.168.43.2'
    user1 = 'rovio'
    password1 = "azwan9669"

    url2 = '192.168.43.3'
    user2 = 'rovio'
    password2 = 'azwan9669'

    app1 = rovioControl('rovio1', url1, user1, password1, rovio_detector, rovio, chaser=True)
    app2 = rovioControl('rovio2', url2, user2, password2, rovio_detector, rovio, chaser=False)









while True:
    print('START!!!')
    while rovio['rovio1_ready'] is False and rovio['rovio2_ready'] is False:
        app1.search_rovio()
        app2.search_rovio()

    app1.backward()
    app2.backward()
    print('both rovio ready!!!!')
    for i in range(3):
        print(i)
        time.sleep(1)


    while app1.main() and app2.main():
        pass

    # t1 = threading.Thread(target=app1.start)
    # t1.start()
    # t2 = threading.Thread(target=app2.start())
    # t1.join()
    app1.toggle_chaser()
    app2.toggle_chaser()


    print('END')







    # If press esc, then head down and stop the loop
    # if app1.key == 27:
    #     app1.rovio.head_down()
    #     break
    #
    # if app2.key == 27:
    #     app2.
