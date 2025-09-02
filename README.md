# Plant Disease Detection Using Deep Learning

Simple Django web app that loads a trained TensorFlow/Keras model to detect plant diseases from leaf images.

This repository contains the Django project, the web UI for uploading images, and a place to drop the trained model artifacts. The repository intentionally does not include the trained model files; add them locally in the `model/` folder (see notes).

## What this project does

- Accepts an uploaded plant leaf image via a web form
- Runs a trained deep learning model to classify disease (or healthy)
- Shows prediction and confidence on the web page

## Requirements

- Python 3.8+ (project tested with Python 3.10/3.11)
- pip
- A trained Keras/TensorFlow model saved with `model.save(...)` or as an HDF5 file

Recommended (optional): create a virtual environment in the repo root.

Windows (cmd.exe):

```bat
python -m venv env
env\Scripts\activate.bat
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install at least:

```bat
pip install django tensorflow pillow numpy
```

## Where to put the model

- Place the model under a directory that is readable by Django, for example:

```text
plant_disease_Detection/app/model/  <-- (ignored by .gitignore)
```

Alternatively, set explicit absolute paths in `plant_disease_Detection/settings.py` by adding two settings:

```python
# Path to the saved model file or folder
MODEL_PATH = r"C:\full\path\to\model"  # or relative path from project root

# Path to class indices JSON mapping file (optional)
CLASS_INDICES_PATH = r"C:\full\path\to\class_indices.json"
```

Make sure the model format is compatible with the Keras/TensorFlow version in `env`.

## Run the app

Activate your virtualenv and run Django's dev server from the repo root:

```bat
env\Scripts\activate.bat
cd plant_disease_Detection
python manage.py runserver
```

Open the local development server address in your browser (for example, 127.0.0.1:8000) and use the upload UI.

## Common troubleshooting

- If you get model loading errors, ensure `MODEL_PATH` points to a valid saved model.
- If predictions look wrong for clearly non-leaf images, the app includes a heuristic to detect non-leaf images and will return a helpful message.

## Git / push readiness

This repo includes a `.gitignore` that excludes model artifacts and other common local files so you can safely commit and push code without large model binaries.

Suggested git workflow on Windows (cmd.exe):

```bat
git init
git add .
git commit -m "Initial commit - app code and README"
# Add a remote if you have one
git remote add origin https://github.com/<your-username>/<your-repo>.git
git branch -M main
git push -u origin main
```

## Notes

- Do not commit model weights, large datasets, or the virtual environment. `.gitignore` provided will exclude common paths like `env/`, `model/`, `*.h5`, and `media/`.
- If you want to store model artifacts in a remote storage, consider using Git LFS or an artifacts server.

If you'd like, I can also generate a `requirements.txt` from your current environment or add a sample `class_indices.json` file.
