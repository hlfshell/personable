import math
import functools
from uuid import uuid4

class Person:

    """
    id = string, uuid for the person. This may change even if its the same person, simply
            because we're looking only at instance id //note- not sure if this will change
            
    pose - last known pose estimation 
    face - boundaries (top, left, bottom, right) of face bounding box
    encodings - [] of known facial encodings
    last_face_scan - how many frames since the last face scan (stops counting after max face scans hit)
    scan_faces_every_n_frames - how often to check faces

    is_visible - is it currently visible
    last_seen - how many frames since the person has been seen last

    pose_index - what index body is it, when we're not checking faces
    
    body_tracking_weights = weights to multiply by when calculating distances
    """

    def __init__(self):
        self.id = uuid4()
        self.last_seen = 0
        self.is_visible = False

        self.pose = []

        self.encodings = []
        self.last_face_scan = None
        
        self.body_tracking_weights = None
        # self.body_tracking_weights = [
        #     0.1, #Nose
        #     1, #Neck
        #     1.5, #RShoulder
        #     0.1, #RElbow
        #     0.1, #RWrist
        #     1.5, #LShoulder
        #     0.1, #LElbow
        #     0.1, #LWrist
        #     1.5, #RHip
        #     0.1, #RKnee
        #     0.1, #RAnkle
        #     1.5, #LHip
        #     0.1, #LKnee
        #     0.1, #LAnkle
        #     0.1, #REye
        #     0.1, #LEye
        #     0.1, #REar
        #     0.1, #LEar
        # ]
    
    def tock(self):
        if not self.is_visible:
            self.last_seen += 1

    def tick(self):
        #reset visiblity
        self.is_visible = False
        self.last_face_scan += 1

    #calculate the face boundaries
    def calculate_face(self):
        headparts = [0, 1, 14, 15, 16, 17]

        left = 1
        right = 0
        top = 1
        bottom = 0

        nose = None
        neck = None

        for body_part in self.pose.body_parts:
            if self.pose.body_parts[body_part].part_idx in headparts:
                # Detect the leftmost, rightmost, bottommost, and topmost positions
                x = self.pose.body_parts[body_part].x
                y = self.pose.body_parts[body_part].y

                if x < left:
                    left = x
                if x > right:
                    right = x
                if y < top:
                    top = y
                if y > bottom:
                    bottom = y

                if body_part == 0:
                    nose = self.pose.body_parts[body_part]
                elif body_part == 1:
                    neck = self.pose.body_parts[body_part]

        if neck is not None and nose is not None:
            nn_distance = (neck.y - nose.y)
            
            top = top - nn_distance
            if top < 0:
                top = 0

        return (top, left, bottom, right)

    #Calculate the distance difference between the two poses given
    def distance_from_pose(self, pose):
        differences = []
        for key, body_part in pose.body_parts.items():
            current_body_part = self.pose.body_parts.get(key)
            if current_body_part is None:
                continue

            difference_x = body_part.x - current_body_part.x
            difference_y = body_part.y - current_body_part.y

            difference_total = math.sqrt( difference_x ** 2 + difference_y **2)
            if self.body_tracking_weights is not None:
                difference_total *= self.body_tracking_weights[key]
            differences.append(difference_total)

        if len(differences) <= 0:
            average_difference = 1
        else:
            average_difference = sum(differences) / len(differences)

        return average_difference


    def update(self, pose):
        self.pose = pose
        self.face = self.calculate_face()
        self.is_visible = True
        self.last_seen = 0

    def set_encoding(self, encoding):
        self.encodings.append(encoding)
        self.last_face_scan = 0
