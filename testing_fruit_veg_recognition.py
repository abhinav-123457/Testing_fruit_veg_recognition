# -*- coding: utf-8 -*-
"""Testing_fruit_veg_recognition.ipynb

# Importing Libraries
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

"""Dataset Link: https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition

# Test set Image Processing
"""

test_set = tf.keras.utils.image_dataset_from_directory(
    '/content/drive/MyDrive/Colab Notebooks/archive/test',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

"""# Loading Model"""

cnn = tf.keras.models.load_model('/content/drive/MyDrive/Colab Notebooks/trained_model.h5')

"""#Visualising and Performing Prediction on Single image"""

#Test Image Visualization
import cv2
image_path = '/content/drive/MyDrive/Colab Notebooks/archive/test/beetroot/Image_2.jpg'
# Reading an image in default mode
img = cv2.imread(image_path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #Converting BGR to RGB
# Displaying the image
plt.imshow(img)
plt.title('Test Image')
plt.xticks([])
plt.yticks([])
plt.show()

"""#Testing Model"""

image = tf.keras.preprocessing.image.load_img(image_path,target_size=(64,64))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = cnn.predict(input_arr)

print(predictions)

# test_set.class_names

result_index = np.argmax(predictions) #Return index of max element
print(result_index)

# Displaying the image
plt.imshow(img)
plt.title('Test Image')
plt.xticks([])
plt.yticks([])
plt.show()

#Single image Prediction
print("It's a {}".format(test_set.class_names[result_index]))
