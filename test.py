import cv2
import matplotlib.pyplot as plt

from classes.Tracker import Tracker

tracker = Tracker()

img = cv2.imread("./test.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("PROCESS START")
tracker.process_frame(img)
print("PROCESS END")
out = tracker.draw_output(img)
# plt.figure()
# plt.suptitle("title")
# plt.imshow(out)
# plt.show()

img = cv2.imread("./test.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("PROCESS START")
tracker.process_frame(img)
print("PROCESS END")
for person in tracker.people:
    print(person.id, " - ", person.is_visible)
out = tracker.draw_output(img)
plt.figure()
plt.suptitle("title")
plt.imshow(out)
plt.show()

# img = cv2.imread("./test2.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# tracker.process_frame(img)
# out = tracker.draw_output(img)
# plt.figure()
# plt.suptitle("title")
# plt.imshow(out)
# plt.show()