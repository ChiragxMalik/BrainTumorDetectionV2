import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import normalize, to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

image_directory = 'datasets/'
INPUT_SIZE = 64

def load_images_from_folder(folder, label_value):
    images = []
    labels = []
    for image_name in os.listdir(folder):
        if image_name.lower().endswith('.jpg'):
            image_path = os.path.join(folder, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                image = Image.fromarray(image, 'RGB')
                image = image.resize((INPUT_SIZE, INPUT_SIZE))
                images.append(np.array(image))
                labels.append(label_value)
    return images, labels

no_images, no_labels = load_images_from_folder(os.path.join(image_directory, 'no'), 0)
yes_images, yes_labels = load_images_from_folder(os.path.join(image_directory, 'yes'), 1)

dataset = np.array(no_images + yes_images)
labels = np.array(no_labels + yes_labels)

x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=0)

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

model = Sequential([
    Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(32, (3, 3), kernel_initializer='he_uniform'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(32, (3, 3), kernel_initializer='he_uniform'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(64),
    Activation('relu'),
    Dropout(0.5),
    Dense(2),
    Activation('softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(
    x_train, y_train,
    batch_size=16,
    verbose=1,
    epochs=10,
    validation_data=(x_test, y_test),
    shuffle=True
)

model.save('BrainTumorDetectionModel.h5')