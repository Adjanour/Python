import cv2
import os

# Directory to save registered faces
dataset_path = "registered_faces"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Initialize the camera
cap = cv2.VideoCapture(0)

user_name = input("Enter your name: ")

# Create a directory for the new user
user_dir = os.path.join(dataset_path, user_name)
if not os.path.exists(user_dir):
    os.makedirs(user_dir)

# Capture 10 images of the user
count = 0
while count < 100:
    ret, frame = cap.read()

    # Display the frame
    cv2.imshow("Registering Face", frame)

    # Save the frame as an image file
    face_path = os.path.join(user_dir, f"{user_name}_{count}.jpg")
    cv2.imwrite(face_path, frame)

    count += 1
    cv2.waitKey(1000)  # Wait 1 second before capturing the next image

cap.release()
cv2.destroyAllWindows()
