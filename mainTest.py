import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('BrainTumorDetectionModel.h5')

image = cv2.imread('E:/Brain Tumor Detection/pred/pred5.jpg')
img = Image.fromarray(image)
img = img.resize((64, 64))
img = np.array(img)
input_img = np.expand_dims(img, axis=0)

predict_x = model.predict(input_img)
result = np.argmax(predict_x, axis=1)
print(result)
