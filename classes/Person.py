import math
import functools
from uuid import uuid4

class Person:

    """
    id = string, uuid for the person. This may change even if its the same person, simply
            because we're looking only at instance id //note- not sure if this will change
    name = string - if provided, it is the matched name based on face. if not, it will generate
                human readable identification (ie A, B, etc)

    pose - last known pose estimation 
    face - boundaries (top, left, bottom, right) of face bounding box
    encodings - [] of known facial encodings

    is_visible - is it currently visible
    last_seen - how many frames since the person has been seen last

    pose_index - what index body is it, when we're not checking faces
    """

    def __init__(self):
        self.id = uuid4()
        self.name = None
        self.last_seen = 0
        self.is_visible = False

        self.pose = []
    
    def tock(self):
        if not self.is_visible:
            self.last_seen += 1

    def tick(self):
        #reset visiblity
        self.is_visible = False

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
        for body_part in pose.body_parts:
            # print("body part", body_part, str(body_part))
            # print(self.pose.body_parts)
            # print(self.pose.body_parts[body_part])
            # print(hasattr(self.pose.body_parts, str(body_part)))
            # print(type(self.pose.body_parts))
            try:
                difference_x = pose.body_parts[body_part].x - self.pose.body_parts[body_part].x
                difference_y = pose.body_parts[body_part].y - self.pose.body_parts[body_part].y

                difference_total = math.sqrt( difference_x ** 2 + difference_y ** 2 )
                differences.append(difference_total)
            except:
                continue
            
            # if hasattr(self.pose.body_parts, str(body_part)):
            #     difference_x = pose.body_parts[body_part].x - self.pose.body_parts[body_part].x
            #     difference_y = pose.body_parts[body_part].y - self.pose.body_parts[body_part].y

            #     difference_total = math.sqrt( difference_x ** 2 + difference_y ** 2 )
            #     differences.append(difference_total)
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
