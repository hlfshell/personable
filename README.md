# personable

`personable` is a module that handles frame by frame tracking of people, with the ability to recognize people via their faces. It utilizes the [face_recognition](https://github.com/ageitgey/face_recognition) and [tf_pose](https://github.com/ildoonet/tf-pose-estimation) libraries.

# Examples
The following examples are available inside this repos, demonstrating the use of the Tracker.

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

Lines 7 and 8 are commented out. If you have a pre-created encodings file, you may uncomment out line 7 in order to have it load the encodings to identify people. If you uncomment out line 8 instead, it will scan and create encodings off of a directory of faces, as `examples/generate_encodings.py` would, prior to activating the webcam.
