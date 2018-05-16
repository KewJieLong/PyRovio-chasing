import cv2


class rovio:
    chase = None
    detector = None
    rovioConrol = None

    def __init__(self, chase, detector, rovioControl):
        self.chase = chase
        self.detector = detector
        self.rovioConrol = rovioControl


    def action(self):
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
                if max_area < area and (width / height > 1.1 and width / height < 1.2):
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
            if (x * frame.shape[0] >= 213 and x * frame.shape[0] <= 426):
                self.rovio.forward()
            elif x * frame.shape[1] > frame.shape[1] / 2:
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


