# Face Recognition Software

This project works by having a folder that contains photos of known faces, then face landmarks are extracted using Google's [Mediapipe](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) and face embeddings are calculated for each known face.
When you run [main.py](https://github.com/TechT3o/face_recognition/blob/master/main.py)  it starts reading frames from your webcam, detects the faces present in each frame and finds a cosine similarity metric between the known face embeddings and the embeddings of the currently found faces. These results are then printed below the face using OpenCV.


All of this functionality is [found in the face_detector.py](https://github.com/TechT3o/face_recognition/blob/master/face_recognizer.py)
In this project the face detection of Mediapipe is used to get the face landmarks 
Then run requirements.txt to download and install the required libraries and then main.py

![face_recognition_demo](https://github.com/TechT3o/face_recognition/assets/87833804/5c941324-0348-43b6-8bc5-e7933dae035d)
