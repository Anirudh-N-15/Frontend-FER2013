from flask import Flask,request,jsonify
import cv2
import numpy as np
import tensorflow as tf
import keras 
from tensorflow import keras
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import LSTMV1, Dense, BatchNormalization

app = Flask(__name__)

model = load_model('face_model.h5', custom_objects={'BatchNormalization': BatchNormalization})

def prepare_image(img):
    img = cv2.resize(img, (48, 48))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = np.reshape(img, (1, 48, 48, 1))  
    return img


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = prepare_image(img)
    prediction = model.predict(img)
    emotion = np.argmax(prediction)  # You may have a list of emotions like ['happy', 'sad', ...]
    
    # Map model output to emotion (example)
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    return jsonify({'emotion': emotions[emotion]})

if __name__ == '__main__':
    app.run(debug=True)    
