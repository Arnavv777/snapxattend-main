import dlib
import numpy as np
import face_recognition_models
from sklearn.svm import SVC
import streamlit as st

from src.database.db import get_all_students


# ---------------------------------------------------
# LOAD DLIB MODELS
# ---------------------------------------------------

@st.cache_resource
def load_dlib_models():

    detector = dlib.get_frontal_face_detector()

    shape_predictor = dlib.shape_predictor(
        face_recognition_models.pose_predictor_model_location()
    )

    face_recognition_model = dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )

    return detector, shape_predictor, face_recognition_model


# ---------------------------------------------------
# GET FACE EMBEDDINGS
# ---------------------------------------------------

def get_face_embeddings(image_np):

    detector, shape_predictor, face_recognition_model = load_dlib_models()

    faces = detector(image_np, 1)

    encodings = []

    for face in faces:

        shape = shape_predictor(image_np, face)

        face_descriptor = face_recognition_model.compute_face_descriptor(
            image_np,
            shape,
            1
        )

        encodings.append(
            np.array(face_descriptor, dtype=float)
        )

    return encodings


# ---------------------------------------------------
# LOAD STUDENTS AND TRAIN CLASSIFIER
# ---------------------------------------------------

@st.cache_resource
def get_trained_model():

    X = []
    y = []

    student_db = get_all_students()

    # No students in database
    if not student_db:
        return None

    for student in student_db:

        embedding = student.get("face_embedding")
        student_id = student.get("student_id")

        if embedding is not None and student_id is not None:

            X.append(
                np.array(embedding, dtype=float)
            )

            y.append(student_id)

    # No valid face embeddings
    if not X:
        return None

    clf = None

    unique_students = set(y)

    # SVC requires at least two different classes
    if len(unique_students) >= 2:

        clf = SVC(
            kernel="linear",
            probability=True,
            class_weight="balanced"
        )

        clf.fit(X, y)

    return {
        "clf": clf,
        "X": X,
        "y": y
    }


# ---------------------------------------------------
# REFRESH MODEL AFTER NEW STUDENT REGISTRATION
# ---------------------------------------------------

def train_classifier():

    st.cache_resource.clear()

    model_data = get_trained_model()

    return model_data is not None


# ---------------------------------------------------
# FACE RECOGNITION
# ---------------------------------------------------

def predict_attendance(class_image_np):

    # Get faces from captured image
    encodings = get_face_embeddings(class_image_np)

    detected_student = {}

    # No face found
    if not encodings:
        return detected_student, [], 0

    # Load registered student embeddings
    model_data = get_trained_model()

    if not model_data:
        return detected_student, [], len(encodings)

    X_train = model_data["X"]
    y_train = model_data["y"]

    # Safety check
    if not X_train or not y_train:
        return detected_student, [], len(encodings)

    # Unique student IDs
    all_students = list(dict.fromkeys(y_train))

    # Face similarity threshold
    resemblance_threshold = 0.4

    # Check every detected face
    for encoding in encodings:

        # Calculate distance from EVERY stored face
        distances = [
            np.linalg.norm(
                stored_embedding - encoding
            )
            for stored_embedding in X_train
        ]

        # Get closest registered face
        best_index = int(
            np.argmin(distances)
        )

        best_match_score = distances[best_index]

        predicted_id = y_train[best_index]

        # Accept ONLY if the face is sufficiently similar
        if best_match_score <= resemblance_threshold:

            detected_student[predicted_id] = True

    return (
        detected_student,
        all_students,
        len(encodings)
    )