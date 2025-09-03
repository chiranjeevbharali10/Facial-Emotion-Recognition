import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import keras.backend as K
# Add for face detection
import os

# --- Custom loss/metric functions (must match your training script) ---
def focal_loss(gamma=2., alpha=0.25, label_smoothing=0.1):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        if label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=1)
    return focal_loss_fixed

def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

# --- Class names (update if needed) ---
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# --- Preprocessing (match model input) ---
def preprocess_frame(frame, target_size=(224, 224)):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# --- Load Haar Cascade for face detection ---
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# --- Load model ---
MODEL_PATH = 'Check_points/res50(1).h5'  # Update as needed
model = load_model(
    MODEL_PATH,
    custom_objects={
        'focal_loss_fixed': focal_loss(gamma=2., alpha=0.25, label_smoothing=0.1),
        'f1_score': f1_score
    }
)

# --- OpenCV webcam loop ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    print('Frame hash:', hash(frame.tobytes()))  # Debug: check if frames are changing

    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) > 0:
        # Use the largest detected face
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        face_img = frame[y:y+h, x:x+w]
        # Preprocess cropped face
        processed = preprocess_frame(face_img)
        # Draw rectangle on the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    else:
        # If no face detected, use the whole frame (or skip prediction)
        processed = preprocess_frame(frame)

    # Predict
    preds = model.predict(processed)
    print('Raw predictions:', preds)  # Debug: print raw model output
    pred_class = np.argmax(preds[0])
    confidence = preds[0][pred_class]
    pred_label = CLASS_NAMES[pred_class]

    # Display prediction on frame
    display_text = f"{pred_label} ({confidence:.2f})"
    cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # Display all class probabilities
    for i, (cls, prob) in enumerate(zip(CLASS_NAMES, preds[0])):
        text = f"{cls}: {prob:.2f}"
        cv2.putText(frame, text, (10, 80 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Webcam Emotion Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# i need to download all the modules which are 