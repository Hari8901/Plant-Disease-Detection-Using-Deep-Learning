from django.shortcuts import render

import os
import json
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow import keras
from PIL import Image
from django.conf import settings

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load the trained model and class indices once when the server starts
model = None
class_names = None

# Load model and classes using paths defined in settings. Keep failures non-fatal but informative.
try:
    model_path = getattr(settings, 'MODEL_PATH', None)
    class_indices_path = getattr(settings, 'CLASS_INDICES_PATH', None)
    if model_path and os.path.exists(model_path):
        model = keras.models.load_model(model_path)
    else:
        print(f"Model file not found at: {model_path}")

    if class_indices_path and os.path.exists(class_indices_path):
        with open(class_indices_path, 'r', encoding='utf-8') as f:
            class_indices = json.load(f)
        # Invert the dictionary to map indices (int) to class names
        # Expected format in JSON: {"class_name": "index"} or {"0": "class_name"}
        # Try to support both shapes robustly.
        class_names = {}
        # If mapping is name->index, invert it
        try:
            # If values are strings that represent integers
            for k, v in class_indices.items():
                class_names[int(v)] = k
        except Exception:
            # If the JSON is already index->name
            try:
                for k, v in class_indices.items():
                    class_names[int(k)] = v
            except Exception:
                # Fallback: keep raw mapping
                class_names = class_indices
    else:
        print(f"Class indices file not found at: {class_indices_path}")

    if model is not None and class_names:
        print("Model and class indices loaded successfully.")
except Exception as e:
    print(f"Error loading model or class indices: {e}")

def home(request):
    prediction_result = None
    image_url = None

    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        image_url = fs.url(file_path)

        if model and class_names:
            try:
                # Preprocess the uploaded image for the model
                img = Image.open(fs.path(file_path)).convert('RGB')
                # Use the input size the model expects; many examples use 224 or 300.
                img = img.resize((300, 300))
                img_array = keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
                img_array = img_array / 255.0  # Rescale the image

                # Ensure the input matches the model's expected input shape.
                # Some saved models expect a 4D tensor (None, H, W, C).
                # Others were trained on flattened vectors and expect shape (None, N).
                note = None
                try:
                    model_input = getattr(model, 'input_shape', None)
                    # model_input can be like (None, H, W, C) or (None, N)
                    if isinstance(model_input, list):
                        model_input = model_input[0]

                    if model_input and len(model_input) == 4:
                        _, exp_h, exp_w, exp_c = model_input
                        # Only resize/convert if expected dims are concrete
                        if exp_h and exp_w:
                            # If channel mismatch, convert
                            if exp_c and exp_c != img_array.shape[-1]:
                                if exp_c == 1:
                                    # convert to grayscale
                                    pil = Image.open(fs.path(file_path)).convert('L').resize((exp_w, exp_h))
                                    arr = keras.preprocessing.image.img_to_array(pil)
                                    img_array = np.expand_dims(arr, axis=0) / 255.0
                                else:
                                    pil = Image.open(fs.path(file_path)).convert('RGB').resize((exp_w, exp_h))
                                    arr = keras.preprocessing.image.img_to_array(pil)
                                    img_array = np.expand_dims(arr, axis=0) / 255.0
                            else:
                                # resize to expected spatial dims
                                if (img_array.shape[1], img_array.shape[2]) != (exp_h, exp_w):
                                    pil = Image.open(fs.path(file_path)).convert('RGB').resize((exp_w, exp_h))
                                    arr = keras.preprocessing.image.img_to_array(pil)
                                    img_array = np.expand_dims(arr, axis=0) / 255.0
                    elif model_input and len(model_input) == 2:
                        # Model expects a flat vector length N
                        N = int(model_input[1])
                        vec = img_array.reshape(-1)
                        L = vec.size
                        if L != N:
                            note = ''
                            if L < N:
                                # pad with zeros
                                pad = np.zeros((N - L,), dtype=vec.dtype)
                                vec = np.concatenate([vec, pad])
                                note = 'padded'
                            else:
                                # truncate
                                vec = vec[:N]
                                note = 'trimmed'
                        img_array = vec.reshape(1, N)
                except Exception as e:
                    # If anything goes wrong while adapting, continue and let prediction raise a clear error
                    print(f"Warning: failed to adapt input to model shape: {e}")

                # Run prediction and interpret results
                predictions = model.predict(img_array)
                # If model outputs a vector per sample
                if predictions.ndim == 2:
                    probs = predictions[0]
                else:
                    probs = np.ravel(predictions)

                predicted_class_index = int(np.argmax(probs))
                confidence = float(probs[predicted_class_index]) if probs.size > predicted_class_index else 0.0
                predicted_class_name = class_names.get(predicted_class_index, str(predicted_class_index))

                # Decide not-a-leaf based on class name or low confidence.
                not_leaf_names = {"not a leaf", "non-leaf", "background", "other", "unknown"}
                if predicted_class_name and isinstance(predicted_class_name, str) and predicted_class_name.strip().lower() in not_leaf_names:
                    prediction_result = "Uploaded image is not a leaf. Please upload a clear image of a plant leaf."
                elif confidence < 0.45:
                    # Low confidence -> likely not a valid leaf image or ambiguous
                    prediction_result = "Model confidence is low for this image. It may not contain a clear leaf. Please upload a clearer leaf image."
                else:
                    prediction_result = f"Prediction: {predicted_class_name} (confidence: {confidence:.2f})"
            except Exception as e:
                prediction_result = f"An error occurred during prediction: {e}"
        else:
            # Give more helpful error messages depending on what's missing
            if model is None and not class_names:
                prediction_result = "Model and class mappings are not loaded. Check server logs."
            elif model is None:
                prediction_result = "Model is not loaded. Cannot perform prediction."
            else:
                prediction_result = "Class mappings are not loaded. Cannot interpret prediction."

    return render(
        request, "index.html", {"prediction": prediction_result, "image_url": image_url}
    )
