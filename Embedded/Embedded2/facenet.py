import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained FaceNet model
model = load_model('facenet_keras.h5')
print("Loaded Model Successfully")
