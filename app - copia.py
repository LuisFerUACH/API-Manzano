
#Paso 1: Importar todas las dependencias del proyecto

import os
import requests
import numpy as np
import tensorflow as tf
import efficientnet.tfkeras as efn



import os
import time
import pandas as pd
import numpy as np

import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import clear_output
from IPython import display

from sklearn.model_selection import train_test_split
from sklearn import metrics

import tensorflow.keras.layers as L
import efficientnet.tfkeras as efn
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

from tqdm import tqdm as tqdm
import gc

AUTOTUNE = tf.data.experimental.AUTOTUNE


#Hacer el downgrade a la versión 1.1.0 con pip install scipy==1.1.0
from matplotlib.pyplot import imread

from flask import Flask, request, jsonify

print(tf.__version__)

#Paso 2: Cargar el modelo pre entrenado
with open('plant_model.json', 'r') as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)

# cargar los pesos en el modelo
model.load_weights("plant_model.h5")

#model = efn.EfficientNetB7(weights="plant_model.h5", include_top=False)



def load_image(image_id, label=None, image_size=(256, 256)):
    
    if image_id.numpy().decode("utf-8").split('_')[0]=='gan' and len(image_id.numpy().decode("utf-8").split('_'))==2:
        image_id = int(image_id.numpy().decode("utf-8").split('_')[1])
        return md_gan[image_id], [0,1,0,0]
    else:        
        bits = tf.io.read_file(image_id)
        image = tf.image.decode_jpeg(bits, channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, image_size)
        if label is None:
            return image
        else:
            return image, label


def decode_image(filename, label=None, image_size=(256, 256)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    
    if label is None:
        return image
    else:
        return image, label
    
    # normalizing the images to [-1, 1]
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image

def random_jitter(image):
    # resizing to 256 x 256 x 3
    image = tf.image.resize(image, [256,256],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # random mirroring
    image = tf.image.random_flip_left_right(image)

    return image



def preprocess_image_test(image, label):
    image = normalize(image)
    return image





def format_path(st):
    GCS_DS_PATH = './uploads/' + st + '.jpg'
    return GCS_DS_PATH 
train_data = pd.read_csv('./test.csv')
test_paths = train_data.image_id.apply(format_path).values


print(test_paths)
test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_paths,)
    .map(decode_image, num_parallel_calls=AUTOTUNE)
    .batch(1)
)


sub = pd.read_csv("./sample_submission.csv")

probs = model.predict(test_dataset)


sub.loc[:, 'healthy':] = probs
sub.to_csv('submission.csv', index=False)
sub.head()
#Paso 3: Crear la API con Flask
#Crear una aplicación de Flask
app = Flask(__name__)

#Definir la función de clasificación de imágenes
@app.route("/api/v1/<string:img_name>", methods=["POST"])
def classify_image(img_name):
    #Definir donde se encuentra la carpeta de imágenes
    upload_dir = "uploads/"
    #Cargar una de las imágenes de la carpeta
    image = imread(upload_dir + img_name)
    
    #Definir la lista de posibles clases de la imagen
    classes = ["healthy", "multiple_diseases", "rust", "scab"]
    
    #Hacer la predicción utilizando el modelo pre entrenado
    #prediction = model.predict([test_dataset])

    #Devolver la predicción al usuario
    res = sub.loc[sub['image_id'] == img_name.removesuffix('.jpg')].loc[0].drop(['image_id'])


    
    idMax = int(np.argmax(res))
    category = res.index[idMax]
    
    print(idMax)
    
    #result = str(category) + ' = ' + str(res[idMax])
    

    result = {
  category : res[idMax]
  }
    
    return result

#Iniciar la aplicación de Flask
app.run(port=5000, debug=False)
