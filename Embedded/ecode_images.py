import dlib
import cv2
import numpy as np
import os
import pickle

# Load Dlib models
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

known_faces = {}

# Encode faces from the registered dataset
for user_name in os.listdir("registered_faces"):
    user_dir = os.path.join("registered_faces", user_name)
    encodings = []
    for img_name in os.listdir(user_dir):
        img_path = os.path.join(user_dir, img_name)
        img = cv2.imread(img_path)
        detected_faces = detector(img, 1)
        for d in detected_faces:
            shape = sp(img, d)
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            encodings.append(np.array(face_descriptor))

    known_faces[user_name] = encodings

# Save the encodings to a file
with open("face_encodings.pkl", "wb") as f:
    pickle.dump(known_faces, f)
