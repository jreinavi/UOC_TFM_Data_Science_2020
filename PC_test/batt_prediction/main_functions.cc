/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/examples/batt_prediction/main_functions.h"
#include "tensorflow/lite/micro/examples/batt_prediction/model.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/examples/batt_prediction/input_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include <stdio.h>


// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
int inference_count = 0;

const int kTensorArenaSize = 70*1024;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace


// The name of this function is important for Arduino compatibility.
void setup() {
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  static tflite::MicroMutableOpResolver<8> micro_op_resolver;
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddRelu();
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddDequantize();

  // Operation declaration for older versions of Tensorflow Lite libraries  
/*  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_QUANTIZE,
                               tflite::ops::micro::Register_QUANTIZE());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RELU,
                               tflite::ops::micro::Register_RELU());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                               tflite::ops::micro::Register_RESHAPE());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_DEQUANTIZE,
                               tflite::ops::micro::Register_DEQUANTIZE()); */


   // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Obtain pointer to the model's input tensor.
  input = interpreter->input(0);

  //Serial port initialization for microcontroller
  //Serial.begin(9600);
}

// The name of this function is important for Arduino compatibility.
void loop() {
  
  // Values of mean and std for reconversion
  float bat_mean = 61.166389;
  float bat_std = 23.367817;
 
  // Place the input in the model's input tensor (input_data.h)
  for (int i = 0; i < 24; i++){
      input->data.f[i] = x[inference_count][i];
      }

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();

  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on x: %f\n",
                         static_cast<double>(x[0][0]));
    return;
  }

  // Obtain the output from model's output tensor and denormalize
    float y_pred = interpreter->output(0)->data.f[0];
    float y = (y_pred * bat_std) + bat_mean;

  // Print predicted value and also real data an prediction during model training
    float y_real = (y_test[inference_count]*bat_std) + bat_mean;
    float y_real_python = (y_python[inference_count]*bat_std) + bat_mean;

    printf("[%d] Carga de batería --> inferencia: %f - inferencia script: %f - real: %f\n", 
            inference_count, static_cast<double>(y), static_cast<double>(y_real_python), 
            static_cast<double>(y_real));

  // Output the result, through serial port
    //Serial.print("predicted: ");
    // Serial.print(y);
    //Serial.println(y);

  // Increment the inference_counter, and reset it if we have reached
  // the total number of data per cycle
  inference_count += 1;
  if (inference_count >= 288) inference_count = 0;
  sleep(2);
  //delay(2000);
}
