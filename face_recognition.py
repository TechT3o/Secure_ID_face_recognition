import os

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import matplotlib.pyplot as plt
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time


def draw_face_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

def draw_body_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()


def create_embedding_from_landmarks(landmarks):
    """
    Creates a flat embedding vector from facial landmark coordinates.

    Parameters:
    - landmarks: An iterable of facial landmarks where each landmark
                 is an object with 'x' and 'y' attributes (e.g., a list of tuples or a complex object).

    Returns:
    - A numpy array representing the flattened embedding of the landmarks.
    """

    # Flatten the landmark coordinates into a single vector
    embedding = []
    for landmark in landmarks:
        embedding.extend([landmark.x, landmark.y])

    return np.array(embedding)


def cosine_similarity(embedding1, embedding2):
    """
    Calculate the cosine similarity between two embeddings.

    Parameters:
    - embedding1: A numpy array representing the first embedding.
    - embedding2: A numpy array representing the second embedding.

    Returns:
    - The cosine similarity between the two embeddings.
    """

    # Calculate the dot product of the two vectors
    dot_product = np.dot(embedding1, embedding2)

    # Calculate the magnitude (norm) of each vector
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    # Calculate the cosine similarity
    similarity = dot_product / (norm1 * norm2)

    return similarity

def get_known_face_embeddings(known_face_path, face_detector):
    known_embeddings = []
    for file in os.listdir(known_face_path):
        if ".png" in file or ".jpg" in file:
            img = cv2.imread(os.path.join(known_face_path, file))
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            face_detection_result = face_detector.detect(image)
            face_landmarks = face_detection_result.face_landmarks
            if len(face_landmarks) > 0:
                known_embeddings.append(create_embedding_from_landmarks(face_landmarks[0]))
            else:
                print(f"No face in {file}")
    return known_embeddings

# STEP 2: Create an FaceLandmarker object.
face_base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
face_options = vision.FaceLandmarkerOptions(base_options=face_base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

known_embeddings = get_known_face_embeddings("ref_img", face_detector=face_detector)

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Webcam Unavailable")
        break

    start_time = time.time()

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    face_detection_result = face_detector.detect(image)

    face_annotated_image = draw_face_landmarks_on_image(image.numpy_view(), face_detection_result)
    fully_annotated_image = face_annotated_image

    face_landmarks = face_detection_result.face_landmarks
    # landmarks = [[landmark.x, landmark.y] for landmark in face_landmarks[0]]
    if len(face_landmarks) > 0:
        embedding = create_embedding_from_landmarks(face_landmarks[0])

        similarities = [cosine_similarity(known_embedding, embedding) for known_embedding in known_embeddings]
        print(similarities, np.argmin(similarities))

        # Show current FPS
        cv2.putText(fully_annotated_image, f"FPS: {round(1 / (time.time() - start_time), 2)}", (5, 20),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.3, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Face Recognition', fully_annotated_image)

    # Hit 'q' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()