import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "brain_tumor_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

def predict_image(image_path):
    image = load_img(
        image_path, 
        target_size=(224, 224),
        color_mode='rgb'
    )

    image_array = img_to_array(image).astype('float32')
    image_array = image_array / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    scores = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(scores))

    return {
        "predicted_class" : class_names[predicted_index],
        "probability" : float(scores[predicted_index]),
        "all_scores" : {
            name: float(score)
            for name, score in zip(class_names, scores)
        } 
    }