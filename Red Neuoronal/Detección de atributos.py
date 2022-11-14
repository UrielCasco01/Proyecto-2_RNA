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
from tensorflow.keras.optimizers import RMSprop, SGD

#############
#Carga de datos#
#atributos = 'Base de datos/list_attr_celeba.txt'
#atributos_modificado = 'C:/Users/wwwle/Documents/Materias_Universidad/Redes Neuronales/Proyecto-2_RNA/Base de datos/list_attr_celeba_modificado.txt'
#with open(atributos, 'r') as f:
    #print("skipping: " + f.readline())
    #print("skipping headers: " + f.readline())
    #with open(atributos_modificado, 'w') as newf:
        #for line in f:
            #new_line = ' '.join(line.split())
            #newf.write(new_line)
            #newf.write('\n')
            
#Se definen los parámetros a utilizar
epochs = 10
batch_size = 40
optimizer = 'rmsprop'
ih, iw = 192, 192 #tamaño de la imagen
input_shape = (ih, iw,3)

#Se define el dataframe
atributos = 'C:/Users/warri/Downloads/Proyecto-2_RNA-main/Base de datos/list_attr_celeba_modificado.txt'
df = pd.read_csv(atributos, sep=' ',  header=None, engine='python')
#skipfooter=135066,
#Se separan las imagenes y sus atributos para poder modificar los valores de -1 a 0, luego se vuelven a unir
files = tf.data.Dataset.from_tensor_slices(df[0])
attributes= tf.data.Dataset.from_tensor_slices(df.iloc[:,1:].to_numpy().astype('int64')).map(lambda x: ((x+1)/2))
data = tf.data.Dataset.zip((files,attributes))
ruta_imagenes = 'C:/Users/warri/Downloads/Proyecto-2_RNA-main/Base de datos/img_align_celeba/'
def process_file(file_name, attributes):
    image = tf.io.read_file(ruta_imagenes + file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [ih,iw])
    image /= 255.0
    return image, attributes
imagen_etiquetada = data.map(process_file).batch(batch_size)

#Se definen parámetros de la red y se dividen los datos en datos de entrenamiento y prueba
num_train = int(len(df)*0.7)
num_test =len(df) - num_train
epochs_steps = num_train // batch_size
test_steps = num_test // batch_size
data_train = imagen_etiquetada.take(num_train)
data_test = imagen_etiquetada.skip(num_train)
########################################
wandb.init(project="reconocimiento facial")
wandb.config.epochs = epochs
wandb.config.batch_size = batch_size
wandb.config.optimizer = optimizer
#######################################
#Se comienza a definir a estructura de la red neuronal
model = Sequential()
#Primera capa (Convolucional)
model.add(Conv2D(40, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#Segunda capa (Convolucional)
model.add(Conv2D(80, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#Tercera capa (Convolucional)
model.add(Conv2D(120, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#Cuarta capa (Plana)
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))
 #
model.add(Dense(40))
model.add(Activation('sigmoid'))
##########################################
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['binary_accuracy'])
#
model.summary()
#
history= model.fit(data_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=data_test,
    validation_steps=test_steps,
    callbacks=[WandbCallback()])

model.save('rfnn.h5')
