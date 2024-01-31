import face_recognition
import cv2
import numpy as np
import os
import time
# import tkinter
# from tkinter import filedialog


class FaceRecognizer:
    _ID_FOLDER: str
    _REF_FOLDER: str
    _video_capture: cv2.VideoCapture
    _known_faces_encodings: list
    _known_faces_names: list

    def __init__(self, id_folder: str = 'id_img', ref_folder: str = 'ref_img'):
        self.ID_FOLDER = id_folder
        # tkinter.Tk().withdraw()
        # self.ID_FOLDER = filedialog.askdirectory()
        self.REF_FOLDER = ref_folder

        if not os.path.exists(self.REF_FOLDER):
            os.makedirs(self.REF_FOLDER)

        self.video_capture = cv2.VideoCapture(0)

        self.known_faces_encodings, self.known_faces_names = [], []

        # Extract faces from id cards and loads the faces in the known faces encoding
        self.extract_faces_from_ids()
        self.load_faces_from_folder()

    def extract_faces_from_ids(self) -> None:
        """
        Finds faces in ID cards stored in the ID_FOLDER and extracts the image part in a .jpg in the REF_FOLDER
        :return: None
        """
        for filename in os.listdir(self.ID_FOLDER):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image = cv2.imread(os.path.join(self.ID_FOLDER, filename))
                found_faces = face_recognition.face_locations(image)
                if len(found_faces) > 0:
                    top, right, bottom, left = found_faces[0]
                    cv2.imwrite(os.path.join(self.REF_FOLDER, f"face_{filename.split('.')[0]}.jpg"),
                                image[top:bottom, left:right])
                else:
                    print(f"No faces found for {filename}")

    def load_faces_from_folder(self) -> None:
        """
        Loads faces from REF_FOLDER and extracts the face encodings and the respective person's name
        :return: None
        """
        for filename in os.listdir(self.REF_FOLDER):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image = face_recognition.load_image_file(os.path.join(self.REF_FOLDER, filename))
                encoding = face_recognition.face_encodings(image)[0]
                self.known_faces_encodings.append(encoding)
                self.known_faces_names.append(filename.split(".")[0])

    def run(self) -> None:
        """
        Main loop. Reads frame from webcam, finds faces in the frame, obtains the face encoding and compares
        the encoding with the known face encodings. If similar enough face is detected bounding box of face is drawn
        and face_info.txt with is created.
        :return: None
        """
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                print("Webcam Unavailable")
                break

            # keep time to calculate FPS
            start_time = time.time()

            rgb_frame = frame[:, :, ::-1].astype(np.uint8)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(self.known_faces_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(self.known_faces_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_faces_names[best_match_index]

                # Draw rectangle around the face, put name and similarity score
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"{name}, sm{round(face_distances[best_match_index], 2)}",
                            (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0,
                            (255, 255, 255), 1)

                # Get face center and write .txt file
                center_x = (left + right) // 2
                center_y = (top + bottom) // 2
                self.save_center_in_txt(round(face_distances[best_match_index], 2), center_x, center_y)

            # Show current FPS
            cv2.putText(frame, f"FPS: {round(1 / (time.time() - start_time), 2)}", (5, 20),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.3, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Face Recognition', frame)

            # Hit 'q' on the keyboard to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release handle to the webcam
        self.video_capture.release()
        cv2.destroyAllWindows()

    @staticmethod
    def save_center_in_txt(similarity: float, center_x: int, center_y: int) -> None:
        """
        Writes face detection data in face_info.txt in the provided sm={similarity}&kx={center_x}&ky={center_y}&
        format.
        :param similarity: How similar the distance from the detected face embedding is to the face in the database
        :param center_x: Center of face x coordinate
        :param center_y: Center of face y coordinate
        :return: None
        """
        with open('face_info.txt', 'w') as file:
            # Write a line to the file
            file.write(f'sm={similarity}&kx={center_x}&ky={center_y}&')
