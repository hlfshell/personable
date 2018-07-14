import cv2
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import math
import face_recognition
import pickle
import os
from os.path import join, isfile

from .Person import Person

class Tracker:

    """
    attributes:
    
    people = dict of people, key is their uuid
    names = dict of names, key is uuid

    frame_count = count of frames tracked
    scan_every_n_frames = # of frames between person scanning their face
    maximum_difference_to_match = the maximum average differnece of points to 
                                    mark it as not the same pose

    estimator = TfPoseEstimator

    poses = [] of poses from TfPoseEstimator from current frame
    faces = [] positions of faces from current frame
    save_faces_to - None if disabled, a string for an existing dir if enabled
    """

    def __init__(self):
        self.frame_count = 0
        self.scan_every_n_frames = 120
        self.max_face_scans = 5
        self.maximum_difference_to_match = 0.08

        self.names = {}
        self.people = {}

        self.estimator = TfPoseEstimator(get_graph_path("mobilenet_thin"), target_size=(432, 368))
        self.encodings = {}
        self.save_faces_to = None

    def load_encodings(self, filepath):
        self.encodings = pickle.loads(open("./encodings.p", "rb").read())

    def save_encodings(self, filepath):
        encoding_file = open(filepath, "wb")
        encoding_file.write(pickle.dumps(self.encodings))
        encoding_file.close()

    def create_encodings(self, faces_directory):
        facedirs = [filename for filename in os.listdir(faces_directory) if not isfile(join(faces_directory, filename))]
        faces = dict.fromkeys(facedirs, [])
        encodings = {}

        for name in facedirs:
                faces[name] = []

                for filepath in [filename for filename in os.listdir(join(faces_directory, name)) if isfile(join(faces_directory, name, filename))]:
                    faces[name].append(join(faces_directory, name, filepath))

        for name in faces:
            encodings[name] = []
            
            for filepath in faces[name]:
                try:
                    image = cv2.imread(filepath)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    encoding = face_recognition.face_encodings(image, [(0, 0, image.shape[0], image.shape[1])])
                    encodings[name].append(encoding[0])
                except:
                    continue

        self.encodings = encodings

    def get_pose(self, image):
        w = 432
        h = 368
        return self.estimator.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)

    def draw_output(self, image, draw_body=True, draw_face=True, draw_label=True):
        if draw_body:
            poses = []
            for person in self.people:
                if person.is_visible:
                    poses.append(person.pose)

            TfPoseEstimator.draw_humans(image, poses, imgcopy=False)

        for person in self.people:
            if not self.people[person].is_visible:
                continue

            if draw_face:
                top, left, bottom, right = self.people[person].face

                top = math.floor(top * image.shape[0])
                bottom = math.floor(bottom * image.shape[0])
                left = math.floor(left * image.shape[1])
                right = math.floor(right * image.shape[1])

                cv2.rectangle(image, (left, top), (right, bottom), (0,0, 255), 2, 0)

                if draw_label:
                    cv2.putText(image, self.people[person].id, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))


        return image

    def scan_face(self, image, person):
        if not person.is_visible:
            return
        
        top, left, bottom, right = person.face

        top = math.floor(top * image.shape[0])
        left = math.floor(left * image.shape[1])
        bottom = math.floor(bottom * image.shape[0])
        right = math.floor(right * image.shape[1])

        encoding = face_recognition.face_encodings(image, [(top, right, bottom, left)])

        if len(encoding) <= 0:
            return
        
        encoding = encoding[0]
        person.set_encoding(encoding)

    def compare_known_faces(self, person):
        if len(list(self.encodings.keys())) is 0:
            return

        name_key = []
        encodings = []
        counts = {}
        for name in self.encodings:
            results = face_recognition.compare_faces(self.encodings[name], person.encodings[-1])

            match_count = results.count(True)
            counts[name] = match_count

            if match_count / len(results) >= 0.75:
                break

        biggest_match = max(counts, key=counts.get)

        if(counts[biggest_match] <= 3):
            return None
        else:
            return biggest_match

    # Handed a frame to process for tracking
    def process_frame(self, image):
        self.frame_count += 1

        #1 - Generate all the poses
        self.poses = self.get_pose(image)

        #3 - Tick each person
        for person in self.people:
            self.people[person].tick()

        #2 - see if the pose is someone we've seen in our people,
        # or if it's someone new to create a new person object for      
        new_people = []

        for pose in self.poses:
            handled = False
            for person in self.people:
                difference = self.people[person].distance_from_pose(pose)
                if difference < self.maximum_difference_to_match:
                    self.people[person].update(pose)

                    handled = True
                    break

            if handled:
                continue
            else:
                #Create a new person
                person = Person()
                person.update(pose)
                new_people.append(person)

        for person in new_people:
            self.people[person] = person
            
        #4 - Now that we've generated the people, "tock" through all people
        # in order to have their decay occur
        for person in self.people:
            #scan the face of all new people
            if person in new_people:
                self.scan_face(image, person)

            #Scan the face of everyone else that hasnt been scanned for
            #self.scan_every_n_frames

            if person.is_visible and person.last_face_scan % self.scan_every_n_frames == 0 and len(person.encodings) < self.max_face_scans:
                self.scan_face(image, person)

            if person.last_face_scan == 0:
                id = self.compare_known_faces(person)

                older_person = self.people.get(id)
                if older_person is not None:
                    person[id] = person

                if id is not None:
                    person.id = id
                elif self.save_faces_to:
                    person.save_face(image, self.save_faces_to)

            self.people[person].tock()


        
