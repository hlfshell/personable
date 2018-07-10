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
        print("tick", self.id)
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

    #Given pose data and your last pose, is this new pose you?
    #Maybe redo this with a calculated center of mass via torso?
    # Can we just assume that position if we just have neck and shoulders?
    def is_it_you(self, pose):
        #Pay attention to only central torso body parts
        torso_parts = [1, 2, 5, 8, 11]

        distances = []
        for body_part in pose.body_parts:
            if body_part not in torso_parts:
                continue

            #We should have either 1 or 0 matches
            if hasattr(self.pose.body_parts, str(body_part)):
                #Compare total difference
                difference_x = pose.body_parts[body_part].x - self.pose.body_parts[body_part].x
                difference_y = pose.body_parts[body_part].y - self.pose.body_parts[body_part].y

                distance = math.sqrt((difference_x ** 2) + (difference_y ** 2))

                distances.append(distance)
                print("distance", distance)
        
        if len(distances) > 0:
            average_distance = functools.reduce(lambda total, current: total+current, distances) / len(distances)
        else:
            print("no distances")
            #Is this the right fail case?
            average_distance = 0

        print("distance", self.id, average_distance)
        return average_distance < .05



    def update(self, pose):
        self.pose = pose
        self.face = self.calculate_face()
        self.is_visible = True
        self.last_seen = 0
