"""
Main script, put the folder where the ID pictures exist and optionally select
the reference folder where the known faces exist. Then run with a webcam connected
 and see window with hte detected faces.
"""
from face_recognizer import FaceRecognizer

ID_FOLDER = 'id_img'
REF_FOLDER = 'ref_img'

if __name__ == "__main__":
    face_rec = FaceRecognizer(ID_FOLDER, REF_FOLDER)
    face_rec.run()