# UOC_TFM_Data_Science_2020

Repositorio de Trabajo Fin de Máster

## Abstract

This project presents the design of a deep learning model for the prediction of an IoT device energy consumption. Based on the edge computing paradigm, the model may be loaded directly into the device and thus, taking advantage of its processing capacity, obtain predictions of its consumption. For the model training, a history of readings from a IoT rechargeable battery was used. The battery is charge using solar power. To these data, information about light intensity, temperature and humidity was integrated. The results obtained demonstrate that a model, reduced and adapted to the constraints of embedded devices, preserves the same accuracy as the inferences made by a model run in a traditional environment, such as cloud servers or computers.

## Resumen

Este proyecto presenta el diseño de un modelo de aprendizaje profundo (*deep learning*, en inglés) para la predicción del consumo eléctrico de un dispositivo IoT.  En base al paradigma de procesamiento de datos en la frontera (*edge computing*, en inglés), el modelo puede ser cargado directamente en el dispositivo y así, aprovechando su capacidad de procesamiento, obtener predicciones de su consumo. Para el entrenamiento del modelo se utilizó un histórico de lecturas de una batería recargable de un dispositivo IoT. La batería se carga utilizando energía solar.  A estos datos se integraron datos de intensidad de luz, temperatura y humedad.  Los resultados obtenidos demuestran que un modelo, reducido y adaptado a las condiciones restrictivas de los dispositivos embebidos, conserva la misma precisión que las inferencias realizadas por un modelo ejecutado en un entorno tradicional (servidores en la nube u ordenadores).

**Palabras clave:** analisis predictivo,edge computing, consumo electrico.

### Descripción carpetas

`train` : contiene el modelo entrenado en un Notebook de *Google Colab*.  También el conjunto de datos de entrada y ficheros generados al ejecutar el modelo

`PC_test` : prueba de la aplicación que utiliza el modelo ejecutado en un ordenador.  El código está basado en el esquema del ejemplo *hello_world* de TensorFlow Lite for Microcontrollers (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/examples/hello_world)
