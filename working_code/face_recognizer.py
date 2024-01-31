import face_recognition
import cv2
import numpy as np
import os
import time

# class FaceRecognizer:
#     def __init__(self):

def extract_faces_from_IDs(id_folder, output_folder):
    for filename in os.listdir(id_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = cv2.imread(os.path.join(id_folder, filename))
            top, right, bottom, left = face_recognition.face_locations(image)[0]
            cv2.imwrite(os.path.join(output_folder, f"face_{filename.split('.')[0]}.jpg"), image[top:bottom, left:right])


if __name__ == "__main__":
    ID_FOLDER_NAME = "id_img"
    REF_FOLDER_NAME = "ref_img"
    extract_faces_from_IDs(ID_FOLDER_NAME, REF_FOLDER_NAME)