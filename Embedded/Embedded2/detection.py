from mtcnn.mtcnn import MTCNN
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

import numpy as np



# Initialize MTCNN detector
detector = MTCNN()

def detect_face(image):
    faces = detector.detect_faces(image)
    if faces:
        x, y, width, height = faces[0]['box']
        face = image[y:y+height, x:x+width]
        return face, (x, y, width, height)
    return None, None


# Load the pre-trained FaceNet model
model = load_model('facenet_keras-2.h5')
print("Loaded Model Successfully")


def preprocess_face(face, required_size=(160, 160)):
    # Resize the face to the required size for FaceNet
    face = cv2.resize(face, required_size)
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    face = np.expand_dims(face, axis=0)
    return face

def get_embedding(face):
    face = preprocess_face(face)
    embedding = model.predict(face)
    return embedding[0]


# Dictionary to store known faces
known_faces = {}

def register_face(name, image):
    face, _ = detect_face(image)
    if face is not None:
        embedding = get_embedding(face)
        known_faces[name] = embedding
        print(f"Registered {name} successfully.")
    else:
        print("No face detected.")

def recognize_face(image):
    face, _ = detect_face(image)
    if face is None:
        return "No face detected."

    embedding = get_embedding(face)
    min_distance = float('inf')
    identity = "Unknown"

    for name, known_embedding in known_faces.items():
        distance = np.linalg.norm(embedding - known_embedding)
        if distance < min_distance:
            min_distance = distance
            identity = name

    if min_distance > 0.6:
        identity = "Unknown"

    return identity

def access_control(image):
    identity = recognize_face(image)
    if identity != "Unknown":
        print(f"Access Granted: {identity}")
    else:
        print("Access Denied")


# Load images for testing
image_path = "me.jpg"
image = cv2.imread(image_path)

# Register a known face
register_face("Bernard", image)

# Test with a new image
test_image_path = "me2.jpg"
test_image = cv2.imread(test_image_path)

access_control(test_image)
