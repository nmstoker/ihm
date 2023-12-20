# ihm
Simple image histogram match

Trained on a folder with image subfolders named after the label for the subfolder's image source.

## API

It runs under FastAPI.  Install a virtual environment then

pip install fastapi uvicorn python-multipart
pip install joblib
pip install opencv-python scikit-learn     (can probably use opencv-python-headless but untested currently)

Run it with:

uvicorn ihm_api:app --reload