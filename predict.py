import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Load trained model
MODEL_PATH = "model/plant_disease_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# Manual class names (must match your trained model)
class_names = [
"Apple___Apple_scab",
"Apple___Black_rot",
"Apple___Cedar_apple_rust",
"Apple___Healthy",
"Blueberry___Healthy",
"Cherry_(including_sour)___Powdery_mildew",
"Cherry_(including_sour)___Healthy",
"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
"Corn_(maize)___Common_rust_",
"Corn_(maize)___Northern_Leaf_Blight",
"Corn_(maize)___Healthy",
"Grape___Black_rot",
"Grape___Esca_(Black_Measles)",
"Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
"Grape___Healthy",
"Orange___Haunglongbing_(Citrus_greening)",
"Peach___Bacterial_spot",
"Peach___Healthy",
"Pepper,_bell___Bacterial_spot",
"Pepper,_bell___Healthy",
"Potato___Early_blight",
"Potato___Late_blight",
"Potato___Healthy",
"Raspberry___Healthy",
"Soybean___Healthy",
"Squash___Powdery_mildew",
"Strawberry___Leaf_scorch",
"Strawberry___Healthy",
"Tomato___Bacterial_spot",
"Tomato___Early_blight",
"Tomato___Late_blight",
"Tomato___Leaf_Mold",
"Tomato___Septoria_leaf_spot",
"Tomato___Spider_mites Two-spotted_spider_mite",
"Tomato___Target_Spot",
"Tomato___Tomato_Yellow_Leaf_Curl_Virus",
"Tomato___Tomato_mosaic_virus",
"Tomato___Healthy"
]

def predict_disease(img_path):
    """
    Predict disease for a given image path safely.
    """
    if not os.path.exists(img_path):
        return "Error: Image file not found"

    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(224,224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediction
        prediction = model.predict(img_array)

        # Safety check
        if prediction.shape[1] != len(class_names):
            return f"Error: Model output ({prediction.shape[1]}) does not match {len(class_names)} class labels"

        predicted_class = class_names[np.argmax(prediction)]
        return predicted_class

    except Exception as e:
        return f"Prediction Error: {e}"