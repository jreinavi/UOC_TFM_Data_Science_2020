<!-- mdformat off(b/169948621#comment2) -->

# Predición del consumo eléctrico de un dispositivo IoT en el edge computing

El proyecto realiza inferencias sobre el consumo de batería de un dispostivo IoT utliizando un modelo CNN comprimido utilizando Tensorflow Lite Micro.

Las muestras corresponden a lecturas de cada 10 minutos y el modelo predice la lectura de los próximos 10 minutos tomando en cuenta la última hora de lecturas (6 muestras).

Las librerías utilizadas para desarrollar el código se encuentran disponibles en GitHub:
git clone https://github.com/tensorflow/tensorflow.git

Para ejecutar el código pegar la carpeta del proyecto en la ruta:
tensorflow/tensorflow/lite/micro/examples

## Descripción de los ficheros

* `batt_prediction_test.cc`: realiza un test para comprobar que los datos de entrada al modelo coinciden en dimension y en tipo con los datos del tensor de entrada.
* `Ficheros output_hander`: se utilizan solo en la aplicación de test para validar los resultados
* `main_functions.cc` y `main_funcions.h`: código que ejectuta la inferencia
* `model.h` y `model.cc`: posee la información en binario del modelo CNN
* `input_data.h`: contiene los vectores de entrada para la prueba de la aplicación y las predicciones y etiquetas reales del conjunto de test para comparar.  El conjunto de datos de entrada es una matriz 288x24 y las predicciones vectores de 288.
* carpeta `data`: contiene los CSV extraídos del script de entrenamiento del modelo (en el Jupyter notebook).  Contienen la información para declarar los datos de entrada que permiten probar el modelo.  Los ficheros `train_mean` y `train_std` están los valores de media y desviación estándar para cada uno de las variables; en el código se utilizan los valores de *bat* para desnormalizar las inferencias.
* El resto de ficheros son estándares de TensorFlow Lite