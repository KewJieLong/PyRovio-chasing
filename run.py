from lib import Rovio
import cv2
import numpy as np
import winsound, sys, time

from skimage import filter, img_as_ubyte


class rovioControl(object):
    def __init__(self, url, username, password, port=80):
    	# Initialize the robot with username,pw, ip
        self.rovio = Rovio(url, username=username, password=password,
                           port=port)
        self.last = None
        self.key = 0

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
        while flag ==False:
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
            	cv2.imshow("Facedetector",frame)
                cv2.imwrite("face.png", frame)
                flag = True
                winsound.PlaySound('%s.wav' % 'humandetected', winsound.SND_FILENAME)


    #########################################################################################
    #								Floor Finder algorithm									#
    #########################################################################################
    def floor_finder(self):
    	# Capture image
        frame = self.rovio.camera.get_frame()
        # Perform Gaussian Blur to reduce noise
        gaussian = cv2.GaussianBlur(frame,(5,5),0)
        # Perform Canny Edge Detection to detect edges and lines
        edges = cv2.Canny(gaussian,100,200)
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
        h= np.size(im, 0)
        # Get Width
        w= np.size(im, 1)
        y = 0
        line = []
        #################################################
        #    If found edges, then draw rectangle		#
        #################################################
        # Please improve this: Still can improve		
        for j in range(h-1,0,-1):
            for i in range(w):
                if not im[j][i] == 0:
                    y = j
                    break
            cv2.rectangle(frame,(0,y),(w,h),(245,252,0),2)
        cv2.imshow("Zone",frame)
        # Return the height of the zone
        return h-y

    # MAIN FUNCTION TO BE CALLED
    def main(self):
    	# Whenever initialize, raise head to middle
        self.rovio.head_middle()
        # Get frame and show original capture frame
        frame = self.rovio.camera.get_frame()
        cv2.imshow("Original", frame)

        if not isinstance(frame, np.ndarray):
            return
        frame = self.night_vision(frame)
        frame = self.resize(frame)
        frame = cv2.merge([frame, frame, frame])
        frame = self.show_battery(frame)
        #####################################################
        #				Perform Floor floor_finder			#
        #####################################################
        # If safe zone is more than 80 then check for infrared detection
        # TODO: IMPROVE　ＴＨＥ　ＶＡＬＵＥ	
        if self.floor_finder() > 80:
            if (not self.rovio.ir()):
                self.rovio.api.set_ir(1)
            if (not self.rovio.obstacle()):
                self.rovio.forward()
                self.rovio.forward()
                self.rovio.forward()
                self.rovio.forward()
                self.rovio.forward()
                self.rovio.forward()
            else:
                self.rovio.rotate_right(angle=20, speed=2)
        # Rotate right is safe zone is smaller than 80 pixels
        else:
            self.rovio.rotate_right(angle=20, speed=2)

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
            self.rovio.stop()
            self.face_detection()


#############################################################################
#						Main Class and rovioControl							#
#############################################################################
# TODO: Edit the IP address here
if __name__ == "__main__":
    url = '192.168.173.173'
    user = 'myname'
    password = "12345"
    app = rovioControl(url, user, password)
# While Loop to loop the action
while True:
    app.main()
    # If press esc, then head down and stop the loop
    if app.key == 27:
        app.rovio.head_down()
        break