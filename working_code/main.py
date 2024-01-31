import face_recognition
import cv2
import numpy as np
import os
import time

def save_center_in_txt(similarity, center_x, center_y):
    with open('face_info.txt', 'w') as file:
        # Write a line to the file
        file.write(f'sm={similarity}&kx={center_x}&ky={center_y}&')

def load_faces_from_folder(folder):
    faces_encodings = []
    faces_names = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = face_recognition.load_image_file(os.path.join(folder, filename))
            encoding = face_recognition.face_encodings(image)[0]
            faces_encodings.append(encoding)
            faces_names.append(filename.split(".")[0])
    return faces_encodings, faces_names

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Load faces from IDs
faces_encodings, faces_names = load_faces_from_folder("ref_img")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    start_time = time.time()
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(faces_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(faces_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = faces_names[best_match_index]

        # Draw rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} w sm{round(face_distances[best_match_index], 2)}", (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
        center_x = (left+right)//2
        center_y = (top+bottom)//2
        save_center_in_txt(round(face_distances[best_match_index], 2), center_x, center_y)

    cv2.putText(frame, f"FPS: {round(1/(time.time()-start_time),2)}", (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255), 1)
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
