/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <math.h>

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/examples/batt_prediction/model.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include <stdio.h>

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(LoadModelAndPerformInference) {
  // Define the input and the expected output
  float_t x[24] =  {-0.266499, 0.890627, -0.758371, -0.929757, 
                    -0.266499, 0.890627, -0.758371, -0.929757, 
                    -0.424873, 0.890627, -0.758371, -0.908360, 
                    -0.424873, 0.890627, -0.758371, -0.908360, 
                    -0.424873, 0.890627, -0.758371, -0.929757, 
                    -0.424873, 0.678097, -0.758371, -0.929757};

  float_t y_true = -0.929757;

  // Set up logging
  tflite::MicroErrorReporter micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = ::tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(&micro_error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.\n",
                         model->version(), TFLITE_SCHEMA_VERSION);
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
 
  //Define memory size for model
  const int kTensorArenaSize = 70*1024;
  uint8_t tensor_arena[kTensorArenaSize];

  // Build an interpreter to run the model with
  tflite::MicroInterpreter interpreter(model, micro_op_resolver, tensor_arena,
                                       kTensorArenaSize, &micro_error_reporter);
  
  // Allocate memory from the tensor_arena for the model's tensors
  TF_LITE_MICRO_EXPECT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  // Obtain a pointer to the model's input tensor
  TfLiteTensor* input = interpreter.input(0);
   
  // Make sure the input has the properties we expect
  TF_LITE_MICRO_EXPECT_NE(nullptr, input);

  // The property "dims" tells us the tensor's shape. 
  TF_LITE_MICRO_EXPECT_EQ(4, input->dims->size);
  // The value of each element gives the length of the corresponding tensor.
  // We should expect two single element tensors (one is contained within the
  // other).
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(6, input->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(4, input->dims->data[2]);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[3]);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, input->type);

    /*printf("INPUT:\n");
    printf("Dimension: %d\n", input->dims->size);
    printf("First Dimension: %d\n", input->dims->data[0]);
    printf("Rows: %d\n", input->dims->data[1]);
    printf("Columns: %d\n", input->dims->data[2]);
    printf("Channels: %d\n", input->dims->data[3]);
    printf("Input type: %d\n", input->type);*/

    for (int i = 0; i < 24; i++){
      input->data.f[i] = x[i];
      //printf("x Value  : %f\n", static_cast<double>(x[i]));
    }

  // Run the model and check that it succeeds
  TfLiteStatus invoke_status = interpreter.Invoke();
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  // Obtain a pointer to the output tensor and make sure it has the
  // properties we expect. It should be the same as the input tensor.
  TfLiteTensor* output = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, output->type);

  // Obtain the quantized output from model's output tensor
  float_t y_pred = output->data.f[0];

    printf("OUTPUT:\n");
    //printf("Dimension: %d\n", output->dims->size);
    //printf("First Dimension: %d\n", output->dims->data[0]);
    //printf("Rows: %d\n", output->dims->data[1]);
    //printf("Output type: %d\n", output->type);
    printf("***PREDICTEC VALUE***: %f\n", static_cast<double>(y_pred));
    //printf("Predicted value: %f\n", static_cast<double>(y_pred));
    printf("Actual value: %f\n", static_cast<double>(y_true));
    printf("Difference: %f\n", static_cast<double>(y_pred - y_true));
    //printf("Output scale: %f\n", static_cast<double>(output_scale));
    //printf("Output zero point: %d\n", output_zero_point);

  // Check if the output is within a small range of the expected output
  float_t epsilon = 0.5f;
  TF_LITE_MICRO_EXPECT_NEAR(y_true, y_pred, epsilon);

 
}

TF_LITE_MICRO_TESTS_END
