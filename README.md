# personable

`personable` is a module that handles frame by frame tracking of people, with the ability to recognize people via their faces. It utilizes the [face_recognition](https://github.com/ageitgey/face_recognition) and [tf_pose](https://github.com/ildoonet/tf-pose-estimation) libraries.

# Install

Installing is mostly done through pip, with the exception of the `tf_pose` library as it is not published on pip.

## Installing most of the requirements:

To start the installing the module, do the following:

```
$ git clone https://github.com/hlfshell/personable.git
$ cd personable
$ pip install .
```

## Tensorflow

I did not specify tensorflow as a requirement, as it can run with either `tensorflow` or `tensorflow-gpu` modules. The GPU will obviously run faster if available.

## Installing tf_pose

The `tf_pose`'s github page is [here](https://github.com/ildoonet/tf-pose-estimation). The install can be done as such:

```
$ git clone https://www.github.com/ildoonet/tf-openpose
$ cd tf-openpose
$ pip install -r requirements.txt
```

# Simple usage
Here is a simple usage of using `opencv` to open a webcam and process each frame, tracking whomever it sees and memorizing their faces:
```
import cv2
from personable.Tracker import Tracker

tracker = Tracker()

cam = cv2.VideoCapture(0)
ret_val, image = cam.read()

while True:
    ret_val, image = cam.read()
    tracker.process_frame(image)
    out = tracker.draw_output(image)

    cv2.imshow("output", out)

    if cv2.waitKey(1) == 27:
        break
```

# Examples
The following examples are available inside this repos within the examples folder, demonstrating the use of the Tracker.

## generate_encodings.py
`examples/generate_encodings.py` is not only an example, but also a useful tool. It generates using the Tracker to generate and save facial encodings off of a pre-structured data folder. An example of usage:

Assuming a data folder structure of:
```
  faces
  |--persons_name_1
  |---1.jpg
  |---2.jpg
  |---3.jpg
  |--persons_name_2
  |---1.jpg
  |---2.jpg
```
...wherein you have images with a singular person, organized in folders of the person's name, you can execute the following command:
```
python examples/generate_encodings --faces_path ./faces --encodings_file ./encodings.p
```
...to generate an `encondings.p` file that contains an encodings object that can be used in other examples!

## single_image.py
`examples/single_image.py` will take a singular image and process it, then show it. An example of its usage is

```
python examples/single_image.py --image_path ./test.jpg
```

## webcam.py
`examples/webcam.py` will load up your webcam via opencv and process each frame for you.

Lines 7 and 8 are commented out. If you have a pre-created encodings file, you may uncomment out line 7 in order to have it load the encodings to identify people. If you uncomment out line 8 instead, it will scan and create encodings off of a directory of faces, as `examples/generate_encodings.py` would, prior to activating the webcam. Line 17, if uncommented, would save unrecognized faces as images to the given folder for later identification.


# Settings

The following settings can be configured on a `Tracker` object that will enhance or change its behavior.

* `scan_every_n_frames` - 120 - How often on unidentified people, by frames processed, to process their facial encoding again
* `max_face_scans` - 5 - How many times a face should be scanned on an unidenitifed person before we should just accept the fact that we don't know whom it is
* maximum_difference_to_match - 0.08 - The calculated difference required to match a person to a person's position in a prior frame
* save_faces_to - None - if set to a string, the Tracker will attempt to save faces from unknown people to that folder for later identification


# Tracker functions

* `tracker.process_frame(img)` - Given an image `img`, process the frame. This includes incrementing internal frame count, detecting all humans in the image, linking them to humans from a prior frame (if applicable), and facially scanning all new people
* `tracker.draw_output(image, draw_body=True, draw_face=True, draw_label=True)` - Given an image, mark all people, their faces, and labels, according to the passed settings
* `tracker.create_encodings("./faces")` - Given a directory, create encodings based on the faces in there. See `generate_encodings.p` above
* `load_encodings("./encodings.p")` - Load encodings from a pickle file
* `save_encodings("./encodings.p")` - Save the current encodings to a pickle file
