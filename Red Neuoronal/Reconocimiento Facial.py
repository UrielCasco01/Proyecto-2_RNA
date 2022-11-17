import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import RMSprop, SGD

#Se definen los parámetros a utilizar
ih, iw = 192, 192 #tamano de la imagen
input_shape = (ih, iw, 3) #forma de la imagen: alto ancho y numero de canales
epochs = 40
batch_size = 20
num_train = 700
num_test = 300
epoch_steps = num_train // batch_size
test_steps = num_test // batch_size
#optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00001)
optimizer = 'adam'
#Carga de datos
train_dir = 'C:/Users/wwwle/Documents/Materias_Universidad/Redes Neuronales/Bases de datos/Reconocimiento facial/Train'
test_dir = 'C:/Users/wwwle/Documents/Materias_Universidad/Redes Neuronales/Bases de datos/Reconocimiento facial/Test'

datagen = ImageDataGenerator(
        #rotation_range=10, # rotacion
        rescale = 1./255.)  #Normalización de intensidad de pixeles
        #width_shift_range=0.2, # giro horizontal 
        #height_shift_range=0.2, # giro vertical 
        #zoom_range=0.2, # zoom
        #horizontal_flip=True) # volear horizontalmente
        

train = datagen.flow_from_directory(train_dir,
                  target_size=(ih, iw), # resize to this size
                  #color_mode="rgb", # for coloured images
                  batch_size=batch_size, # number of images to extract from folder for every batch
                  class_mode="binary") # classes to predict
                  #seed=2020 # to make the result reproducible
                  
test = datagen.flow_from_directory(test_dir,
                  target_size=(ih, iw), # resize to this size
                  #color_mode="rgb", # for coloured images
                  batch_size=batch_size, # number of images to extract from folder for every batch
                  class_mode="binary") # classes to predict
                  #seed=2020 # to make the result reproducible
                  

#fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15,15))
#for i in range(4):
    #image = next(train_generator)[0].astype('uint8')
    #image = np.squeeze(image)
    #ax[i].imshow(image)
    #ax[i].axis('off')
#################################################
wandb.init(project="Reconocimiento de mi rostro")
wandb.config.epochs = epochs
wandb.config.batch_size = batch_size
wandb.config.optimizer = optimizer
#################################################
modelo_preentrenado = "C:/Users/wwwle/Documents/Materias_Universidad/Redes Neuronales/Proyecto-2_RNA/Red Neuronal/rfnn.h5"   
model = tf.keras.models.load_model(modelo_preentrenado)
model.add(Dense(1,name="Densa_prueba"))
model.add(Activation('sigmoid', name="Act"))
#Se congelan las primeras 3 capas
for layer in model.layers[:9]:
    layer.trainable = False
################################################# 
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['binary_accuracy'])
#################################################
model.fit_generator(
                train,
                epochs=epochs,
                steps_per_epoch=epoch_steps,
                validation_data=test,
                validation_steps=test_steps,
                callbacks=[WandbCallback()]
                )
#################################################
model.save('Reconocimiento Facial.h5')
