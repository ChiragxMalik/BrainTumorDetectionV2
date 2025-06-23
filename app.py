from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
import os

app = Flask(__name__)

model = load_model('BrainTumorDetectionModel.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

def get_className(classNo):
	if classNo==0:
		return "Brain Tumor has not been detected"
	elif classNo==1:
		return "Brain Tumor has been detected"

def getResult(img_path):
    img = cv2.imread(img_path)
    img = Image.fromarray(img, 'RGB')
    img = img.resize((64, 64))
    img = np.array(img)
    input_img = np.expand_dims(img, axis=0)
    predict_x = model.predict(input_img)
    result = np.argmax(predict_x, axis=1)
    return result

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value = getResult(file_path)
        result = get_className(value)
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)