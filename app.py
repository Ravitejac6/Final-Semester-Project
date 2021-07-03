# from __future__ import division, print_function
# coding=utf-8
import sys
import os
import numpy as np
import cv2

from PIL import Image
import tensorflow as tf
from tensorflow import keras
from skimage import transform
# from tensorflow.keras.utils.np_utils import to_categorical

# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)


# Model saved with Keras model.save()
MODEL_PATH = 'CNN_model.h5'

labels = {0: 'Greening', 1: 'Healthy', 2: 'Canker', 3: 'Black Spot', 4: 'Scab'}
# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(128, 128))

    # Preprocessing the image
    np_image = np.array(img).astype('float32')/255
    np_image = transform.resize(np_image, (128, 128, 3))
    print(np_image.shape)
    x = []
    x.append(np_image)
    x = np.array(x)
    preds = model.predict_classes(x)
    print(preds)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print(preds)
        return labels[preds[0]]


if __name__ == '__main__':
    app.run(debug=True)
