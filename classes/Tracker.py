import cv2
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import math
import face_recognition
import pickle

from .Person import Person

class Tracker:

    """
    attributes:
    
    people = dict of people, key is their uuid
    names = dict of names, key is uuid

    frame_count = count of frames tracked
    scan_every_n_frames = # of frames between person scanning their face

    estimator = TfPoseEstimator

    poses = [] of poses from TfPoseEstimator from current frame
    faces = [] positions of faces from current frame
    """

    def __init__(self):
        self.frame_count = 0
        self.scan_every_n_frames = 72
        self.max_face_scans = 5

        self.names = {}
        self.people = {}

        self.estimator = TfPoseEstimator(get_graph_path("mobilenet_thin"), target_size=(432, 368))

    def get_pose(self, image):
        w = 432
        h = 368
        return self.estimator.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)

    def get_encodings(self, image, faces):
        pass

    def compare_encoding(self, encoding):
        pass

    def update_people(self):
        pass

    def draw_output(self, image, draw_body=True, draw_face=True, draw_label=True):
        if draw_body:
            poses = []
            for person in self.people:
                if person.is_visible:
                    poses.append(person.pose)

            # print(poses, len(poses))
            # poses = list(map(lambda person: person.pose if person.is_visible, self.people))
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

                # if draw_label:
                #     label = ""
                #     if self.people[person].name:
                #         label = self.people[person].name
                #     else:
                #         label = self.people[person].id
                #     cv2.putText(image, label, (right, bottom), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))


        return image

    def scan_face(self, image, person):
        if not person.is_visible:
            return
        
        top, left, bottom, right = person.face

        top = math.floor(top * image.shape[0])
        left = math.floor(top * image.shape[1])
        bottom = math.floor(top * image.shape[0])
        right = math.floor(top * image.shape[1])
        print("SCANNING", person.id)
        encoding = face_recognition.face_encodings(image, [(top, right, bottom, left)])
        
        if len(encoding) <= 0:
            return

        person.set_encoding(encoding)

    def compare_known_faces(self, person):
        pass

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
                if difference < .05:
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
                self.compare_known_faces(person)

            self.people[person].tock()


        
