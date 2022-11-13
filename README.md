# Proyecto-2_RNA
## Red Neuronal entrenada para reconocimiento facial.

### Primera parte del proyecto
Para comenzar con el proyecto se descargó la base de datos de rostros y atributos de [celebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), al terminar la descarga se obtuvieron una carpeta en la que se encontraban las imágenes de los rostros y un documento en formato .txt en el cual estaba la lista de atributos en formato csv.
Para realizar la carga de datos se implemetaron las siguientes líneas de código:

![Carga de datos](https://user-images.githubusercontent.com/106125995/201542933-8c6ea226-580d-451d-87cc-5e2ab9425742.jpg)
Donde la variable "atributos representaba al documento de texto de atributos, la variable "atributos_modificado" un documento en blanco en formao .txt donde se iban a guardar los cambios que se le realizaran al documento de atributos original, y el código restante era el encargado de realizar los cambios necesarios al documento original, los cuales consistían en eliminar los diferentes espacios de separación entre cada valor por un solo espacio y eliminar la primera fila del documento original.

Al realizar lo anterior, se comenzó a desarrollar el código encargado de entrenar a una red neuronal capaz de reconocer los atributos de un rostro en base al documento modificado "atributos_modificado".

Primero se importaron las bibliotecas a utilizar en el código.

![Bibliotecas ocupadas](https://user-images.githubusercontent.com/106125995/201543446-8ff14658-4830-4ebe-90fc-5d7965f9d477.jpg)

Para comenzar con el código se definió una variable "df", la cual sería el dataframe que cargaría el documento "atributos_modificado" en una matriz de pandas.

![image](https://user-images.githubusercontent.com/106125995/201543623-3121e37e-ff6a-4d85-baa2-c18dce2e9e2a.png)

Al haber cargado el dataframe se procedió a realizar una modificación e este, la cual consistió en cambiar los valores dados como -1 a 0, así como otros parámetros necesarios como el tamaño de normalización de todas las imágenes, el rango de intensidad de los pixeles, etc. Al tener "df" en el formato deseado, se realizó la conjunción de las imágenes con la lista de atributos en un solo dataset. Todo esto ejecutado por las siguientes líneas de código:

![Proceso de creación del dataset imagen_etiquetada](https://user-images.githubusercontent.com/106125995/201544251-ca3bcea9-bc65-41a4-833d-3bc719a1941b.jpg)

Con las siguientes líneas de código se definieron algunos parámetros de la red, así como se divieron los datos del dataset "imagen_etiquetada", en los datos de entrenamiento (train) y prueba (test) y se definieron los parámetros a medir con ayuda de la plataforma Wandb.

![parametros](https://user-images.githubusercontent.com/106125995/201547868-31db7948-eb2b-495b-ae2a-b6cc848f6528.jpg)

De esta forma, al ya tener los datos cargados y los parámetros de la red bien definidos se comenzó con la estructuración de la red para reconocer atributos de una fotografía de una persona. La red consistió de un modelo secuencial, al cual se añadieron 3 capas convolucionales de 2 dimensiones, de 40, 80 y 120 filtros, de un tamaño de 3x3 con una función de activación relu y un pooling máximo de tamaño de 2x2. Luego de estas capas se añadió una capa plana densa conformada por 64 neuronas, una funión de activación relu y añadido a esto un dropout del 20%. Finalmente, se añadió una capa densa de 40 neuronas (esto debido a que se tienen 40 atributos) con una función de activación sigmoide.

![Estructura red 1](https://user-images.githubusercontent.com/106125995/201549083-38de733e-57cc-4e95-a4ef-7905264075cf.jpg)

Se utilizó una función de costo **"binary_crossentropy"** con un optimizador **"rmsprop"** y como metrica se utilizó una métrica **"binary_accuracy"** debido a que al utilizar la métrica "accuracy" los resultados del entrenamiento resutaban muy malos.
