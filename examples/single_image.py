import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", help="Path to an image for the test")
args = parser.parse_args()

if args.image_path is None:
    print("An image path must be passed with --image_path image.jpg")
    sys.exit()
    
sys.path.append("..")

import cv2
import matplotlib.pyplot as plt

from classes.Tracker import Tracker

tracker = Tracker()

img = cv2.imread(args.image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

tracker.process_frame(img)
out = tracker.draw_output(img)
plt.figure()
plt.suptitle("Output")
plt.imshow(out)
plt.show()