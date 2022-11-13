# Proyecto-2_RNA
Red Neuronal entrenada para reconocimiento facial.

Para comenzar con el proyecto se descargó la base de datos de rostros y atributos de celebA, al terminar la descarga se obtuvieron una carpeta en la que se encontraban las imágenes de los rostros y un documento en formato .txt en el cual estaba la lista de atributos en formato csv.
Para realizar la carga de datos se implemetaron las siguientes líneas de código:
![Carga de datos](https://user-images.githubusercontent.com/106125995/201542933-8c6ea226-580d-451d-87cc-5e2ab9425742.jpg)
Donde la variable "atributos representaba al documento de texto de atributos, la variable "atributos_modificado" un documento en blanco en formao .txt donde se iban a guardar los cambios que se le realizaran al documento de atributos original, y el código restante era el encargado de realizar los cambios necesarios al documento original, los cuales consistían en eliminar los diferentes espacios de separación entre cada valor por un solo espacio y eliminar la primera fila del documento original.

Al realizar lo anterior, se comenzó a desarrollar el código encargado de entrenar a una red neuronal capaz de reconocer los atributos de un rostro en base al documento modificado "atributos_modificado".

Primero se importaron las bibliotecas a utilizar en el código.
![Bibliotecas ocupadas](https://user-images.githubusercontent.com/106125995/201543446-8ff14658-4830-4ebe-90fc-5d7965f9d477.jpg)

Para comenzar con el código se definió una variable "df", la cual sería el dataframe que cargaría el documento "atributos_modificado" en una matriz de pandas.

![image](https://user-images.githubusercontent.com/106125995/201543623-3121e37e-ff6a-4d85-baa2-c18dce2e9e2a.png)
