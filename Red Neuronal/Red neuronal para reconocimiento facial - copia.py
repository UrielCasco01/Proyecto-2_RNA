import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback
import keras
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

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
train, test = train_test_split(df,test_size=0.3)
#TRAIN
files_tr = tf.data.Dataset.from_tensor_slices(train[0])
attributes_tr = tf.data.Dataset.from_tensor_slices(train.iloc[:,1:].to_numpy())
data_tr = tf.data.Dataset.zip((files_tr,attributes_tr))
ruta_imagenes_tr = 'C:/Users/wwwle/Documents/Materias_Universidad/Redes Neuronales/Proyecto-2_RNA/Base de datos/img_align_celeba/'

def process_file_tr(file_name_tr, attributes_tr):
    image_tr = tf.io.read_file(ruta_imagenes_tr + file_name_tr)
    image_tr = tf.image.decode_jpeg(image_tr, channels=3)
    image_tr = tf.image.resize(image_tr, [180,180])
    image_tr /= 255.0
    return image_tr, attributes_tr

imagen_etiquetada_tr = data_tr.map(process_file_tr)
#for image,attri in imagen_etiquetada.take(1):
    #plt.imshow(image)
    #plt.show()
#TEST
files_ts = tf.data.Dataset.from_tensor_slices(test[0])
attributes_ts = tf.data.Dataset.from_tensor_slices(test.iloc[:,1:].to_numpy())
data_ts = tf.data.Dataset.zip((files_ts,attributes_ts))
ruta_imagenes_ts = 'C:/Users/wwwle/Documents/Materias_Universidad/Redes Neuronales/Proyecto-2_RNA/Base de datos/img_align_celeba/'

def process_file_ts(file_name_ts, attributes_ts):
    image_ts = tf.io.read_file(ruta_imagenes_ts + file_name_ts)
    image_ts = tf.image.decode_jpeg(image_ts, channels=3)
    image_ts = tf.image.resize(image_ts, [180,180])
    image_ts /= 255.0
    return image_ts, attributes_ts

imagen_etiquetada_ts = data_ts.map(process_file_ts)
#############
#Se definen los parámetros para tener los logs en Wandb
epochs = 30
batch_size = 120
optimizer = 'rmsprop'
ih, iw = 180, 180 #tamaño de la imagen
N_d = 141819.
N_t = 60780.
steps_per_epoch = N_d // batch_size
test_steps = N_t // batch_size
data_set_tr= imagen_etiquetada_tr.repeat().batch(batch_size)
data_set_ts= imagen_etiquetada_ts.repeat().batch(batch_size)
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
model.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['accuracy'])
#
history= model.fit(data_set_tr,
    batch_size=batch_size,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=data_set_ts,
    validation_steps=test_steps,
    callbacks=[WandbCallback()])

model.save('rfnn.h5')
       