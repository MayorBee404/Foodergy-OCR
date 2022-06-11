import flask
import io
import string
import time
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, jsonify, request


app = Flask(__name__)

model = tf.keras.models.load_model('ocr_dr.tflite')

def prepare_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img


def predict_result(img):
    return 1 if model.predict(img)[0][0] > 0.5 else 0


@app.route('/', methods=['GET', 'POST'])
def welcome():
    return "Hello Flask!"

@app.route('/predict', methods=['POST'])
def predict():
    request_json = request.json
    print(request_json)
    
    prediction= model.predict(request_json.get('image'))

    return jsonify(prediction)

if __name__ == "__main__":
  app.run(debug=True, host='0.0.0.0', port='80')