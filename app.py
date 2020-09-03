from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image 

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

#specific imports
#from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import cv2

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()

# Load your trained model
Amodel = load_model('models/Amodel.h5')
#head = load_model('models/base.h5')
Gmodel = load_model('models/Gmodel.h5')
#model._make_predict_function()          # Necessary
from keras_vggface.vggface import VGGFace
head = VGGFace(model='vgg16', include_top=False,input_shape=(200, 200, 3))
#model.save('models/base.h5')
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')

def extract_face(filename, required_size=(200, 200)):
      # load image from file
    pixels = cv2.imread(filename)

	# Face detection
    detector = MTCNN()
    results = detector.detect_faces(pixels)
	# Cropping
    x1, y1, width, height = results[0]['box']
    x1, y1 = x1 - 10, y1 + 10
    x2, y2 = x1 + int(1.2*width)+10, y1 + int(1*height) + 5
    face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
    im = Image.fromarray(face)
    im = im.resize(required_size)
    face_array = np.array(im)
    #print(face_array.reshape(1,200,200,3).shape)
    return face_array.reshape(1,200,200,3)



def model_predict(img_path):

    im =  extract_face(img_path)

    features = head.predict(im)
    age = Amodel.predict(features)
    g = Gmodel.predict(features).argmax(axis=-1)
    gender = 'Male' if g==0 else 'Female'
    preds = {
        'Age' : str(np.squeeze(age)),
        'Gender': str(np.squeeze(gender)), 
    }
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
        preds = model_predict(file_path)

        # Expressing outpout as readable form
        result = '\nAge : ' + preds['Age'] + '\nGender : ' + preds['Gender']               # Convert to string
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

