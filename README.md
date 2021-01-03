# UOC_TFM_Data_Science_2020

Repositorio de Trabajo Fin de Máster

## Abstract

This project presents the design of a deep learning model for the prediction of an IoT deviceenergy consumption. 
Based on the edge computing paradigm, the model may be loaded directlyinto the device and thus, taking advantage 
of its processing capacity, obtain predictions of itsconsumption. For model training, a battery charge and discharge 
readings history of a devicepowered by solar panels was used. To these data, information about light intensity,
temperatureand humidity was integrated. The results obtained demonstrate that a reduced model adaptedto the restrictive 
conditions of embedded devices preserves the same precision as the inferencesmade by a model run in a traditional environment, 
such as cloud servers or computers.

## Resumen

Este proyecto presenta el dise ̃no de un modelo de aprendizaje profundo (deep  learning, eningl ́es) para la prediccion 
del consumo el ́ectrico de un dispositivo IoT. En base al paradigma deledge computing, el modelo puede ser cargado directamente 
en el dispositivo y as ́ı, aprovechandosu capacidad de procesamiento obtener predicciones de su consumo. Para el entrenamiento 
del modelo se utiliz ́o un hist ́orico de lecturas de carga y descarga de la bater ́ıa de un dispositivoalimentada por paneles solares.
A estos datos se integrar ́an otros como intensidad de luz, tempe-ratura y humedad. Los resultados obtenidos de demuestran que un 
modelo reducido y adaptadoa las condiciones restrictivas de los dispositivos embebidos conserva la misma precisi ́on que las 
inferencias realizadas por un modelo ejecutado en un entorno tradicional (servidores en la nubeu ordenadores).

**Palabras clave:** an ́alisis predictivo,edge computing, consumo el ́ectrico.

### Descripción carpetas

`train` : contiene el modelo entrenado en un Notebook de *Google Colab*.  También el conjunto de datos de entrada y ficheros generados al ejecutar el modelo

`PC_test` : prueba de la aplicación que utiliza el modelo ejecutado en un ordenador.  El código está basado en el esquema del ejemplo *hello_world* de TensorFlow Lite for Microcontrollers (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/examples/hello_world)
