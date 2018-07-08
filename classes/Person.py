import math
import functools

class Person:

    """
    id = string, uuid for the person. This may change even if its the same person, simply
            because we're looking only at instance id //note- not sure if this will change
    name = string - if provided, it is the matched name based on face. if not, it will generate
                human readable identification (ie A, B, etc)

    pose - last known pose estimation 
    encodings - [] of known facial encodings

    is_visible - is it currently visible
    last_seen - how many frames since the person has been seen last

    pose_index - what index body is it, when we're not checking faces
    """

    def __init__(self):
        self.id = "id"
        self.name = None
        self.last_seen = 0

        self.pose = []

        # self.pose_index = None
    
    #increment lastSeen
    def tick(self):
        self.last_seen += 1

    #Given pose data and your last pose, is this new pose you?
    def is_it_you(self, pose):
        #Pay attention to only central torso body parts
        torso_parts = [1, 2, 5, 8, 11]

        distances = []
        for body_part in pose:
            if body_part not in torso_parts:
                continue

            #First, find the equivalent body part
            body_part_matches = filter(lambda body_part: body_part.part_idx == body_part, self.pose)

            #We should have either 1 or 0 matches
            if len(body_part_matches) > 0:
                match = body_part_matches[0]

                #Compare total difference
                difference_x = pose[body_part].x - match.x
                difference_y = pose[body_part].y - match.y

                distance = math.sqrt((difference_x ** 2) + (difference_y ** 2))

                distances.append(distance)
        
        if len(distances) > 0:
            average_distance = functools.reduce(lambda total, current: total+current, distances) / len(distances)
        else:
            #Is this the right fail case?
            average_distance = 0

        return average_distance < .1

