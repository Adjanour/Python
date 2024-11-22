import face_recognition
import cv2
import numpy

def real_time_face_recognition(known_image_path):
    # Load the known image and learn how to recognize it
    known_image = face_recognition.load_image_file(known_image_path)
    known_face_encodings = face_recognition.face_encodings(known_image)[0]

    # Access video from the webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (OpenCV uses) to RGB color (face_recognition uses)
        rgb_small_frame = numpy.ascontiguousarray(small_frame[:, :, ::-1])

        # Find all face locations and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding in face_encodings:
                # Check if the face is a match for the known face(s)
                matches = face_recognition.compare_faces([known_face_encodings], face_encoding)

                if True in matches:
                    print("Face recognized")

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

# Provide the path to the known image
known_image_path = "me.jpg"
real_time_face_recognition(known_image_path)
