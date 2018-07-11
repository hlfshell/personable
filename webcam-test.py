import cv2
import matplotlib.pyplot as plt

from classes.Tracker import Tracker

tracker = Tracker()

cam = cv2.VideoCapture(0)
ret_val, image = cam.read()

while True:
    ret_val, image = cam.read()
    # cv2.imshow("output", image)

    tracker.process_frame(image)
    out = tracker.draw_output(image)

    cv2.imshow("output", out)

    if cv2.waitKey(1) == 27:
        break