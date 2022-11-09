import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
import wandb
from wandb.keras import WandbCallback
import keras
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import RMSprop

#############
#Carga de datos#
#atributos = 'Base de datos/list_attr_celeba.txt'
atributos_modificado = 'C:/Users/wwwle/Documents/Materias_Universidad/Redes Neuronales/Proyecto-2_RNA/Base de datos/list_attr_celeba_modificado.txt'
#with open(atributos, 'r') as f:
    #print("skipping: " + f.readline())
    #print("skipping headers: " + f.readline())
    #with open(atributos_modificado, 'w') as newf:
        #for line in f:
            #new_line = ' '.join(line.split())
            #newf.write(new_line)
            #newf.write('\n')

df = pd.read_csv(atributos_modificado, sep=' ', header=None)
#print(df.shape)
#print (df.head())

files = tf.data.Dataset.from_tensor_slices(df[0])
attributes = tf.data.Dataset.from_tensor_slices(df.iloc[:,1:].to_numpy())
data = tf.data.Dataset.zip((files,attributes))
ruta_imagenes = 'C:/Users/wwwle/Documents/Materias_Universidad/Redes Neuronales/Proyecto-2_RNA/Base de datos/img_align_celeba/'

def process_file(file_name, attributes):
    image = tf.io.read_file(ruta_imagenes + file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [180,180])
    image /= 255.0
    return image, attributes

imagen_etiquetada = data.map(process_file)
#for image,attri in imagen_etiquetada.take(1):
    #plt.imshow(image)
    #plt.show()


#############
#Se definen los parámetros para tener los logs en Wandb
epochs = 30
batch_size = 120
optimizer = 'rmsprop'
ih, iw = 180, 180 #tamaño de la imagen
N_d = 202599
steps_per_epoch = N_d // batch_size
data_set= imagen_etiquetada.repeat().batch(batch_size)
wandb.init(project="reconocimiento facial")
wandb.config.epochs = epochs
wandb.config.batch_size = batch_size
wandb.config.optimizer = optimizer
wandb.config.hight = ih
wandb.config.width = iw
#############
#Se comienza a definir a estructura de la red neuronal
model = Sequential()
#Primera capa (Convolucional)
model.add(Conv2D(40, (3, 3), input_shape= (ih, iw, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#Segunda capa (Convolucional)
model.add(Conv2D(20, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#Tercera capa (Convolucional)
model.add(Conv2D(10, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#Cuarta capa (Plana)
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))
 #
model.add(Dense(1))
model.add(Activation('tanh'))
#
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
#
history= model.fit(data_set,
    batch_size=batch_size,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch)
    #validation_data=imagen_etiquetada,
    #callbacks=[WandbCallback()])

model.save('rfnn.h5')
       