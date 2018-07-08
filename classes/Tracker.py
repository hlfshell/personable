import cv2
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import math
import face_recognition
import pickle

class Tracker:

    """
    attributes:
    
    people = dict of people, key is their uuid
    names = dict of names, key is uuid

    frame_count = count of frames tracked

    estimator = TfPoseEstimator

    poses = [] of poses from TfPoseEstimator from current frame
    faces = [] positions of faces from current frame
    """

    def __init__(self, pairs):
        self.frame_count = 0

        self.names = {}
        self.people = {}

        self.estimator = TfPoseEstimator(get_graph_path("mobilenet_thin"), target_size=(432, 368))

    def get_pose(self, image):
        w = 432
        h = 368
        return self.estimator.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)

    def get_faces(self, image):

        headparts = [0, 1, 14, 15, 16, 17]

        faces = []

        for pose in self.poses:

            left = image.shape[1]
            right = 0
            top = image.shape[0]
            bottom = 0

            nose = None
            neck = None

            for body_part in pose.body_parts:
                if pose.body_parts[body_part].part_idx in headparts:
                    # Detect the leftmost, rightmost, bottommost, and topmost positions
                    x = math.floor(pose.body_parts[body_part].x * image.shape[1])
                    y = math.floor(pose.body_parts[body_part].y * image.shape[0])

                    if x < left:
                        left = x
                    if x > right:
                        right = x
                    if y < top:
                        top = y
                    if y > bottom:
                        bottom = y

                    if body_part == 0:
                        nose = pose.body_parts[body_part]
                    elif body_part == 1:
                        neck = pose.body_parts[body_part]

            if neck is not None and nose is not None:
                nn_distance = math.floor((neck.y - nose.y) * image.shape[0])
                
                top = top - nn_distance
                if top < 0:
                    top = 0

            faces.append([top, left, bottom, right])

        return faces


    def get_encodings(self, image, faces):
        pass

    def compare_encoding(self, encoding):
        pass

    def update_people(self):
        pass

    def draw_output(self, image):
        pass

    # Handed a frame to process for tracking
    def process_frame(self, image):
        self.frame_count += 1

        #1 - Generate all the poses
        self.poses = self.get_pose(image)

        #2 - see if the pose is someone we've seen in our people,
        # or if it's someone new to create a new person object for

        #3 - For each of the people, generate the faces for that person
        # Note - do this only every couple of frames to keep frame rate up
        self.faces = self.get_faces(image)

        #4 - Now that we've generated the people, tick through all people
        # in order to have their decay occur

        
