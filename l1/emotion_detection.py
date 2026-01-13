import cv2
import numpy as np
import tensorflow as tf

EMOTIONS = ["angry","disgust","fear","happy","sad","surprise","neutral"]

# Load ONCE when imported
emotion_model = tf.keras.models.load_model("emotion_efficientnet.h5")

def predict_emotion_from_crop(face_crop):
    if face_crop is None or face_crop.size == 0:
        return "unknown"

    try:
        img = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        preds = emotion_model.predict(img, verbose=0)
        return EMOTIONS[np.argmax(preds)]
    except Exception:
        return "unknown"