batt_prediction_TEST_SRCS := \
tensorflow/lite/micro/examples/batt_prediction/batt_prediction_test.cc \
tensorflow/lite/micro/examples/batt_prediction/model.cc

batt_prediction_TEST_HDRS := \
tensorflow/lite/micro/examples/batt_prediction/model.h

OUTPUT_HANDLER_TEST_SRCS := \
tensorflow/lite/micro/examples/batt_prediction/output_handler_test.cc \
tensorflow/lite/micro/examples/batt_prediction/output_handler.cc

OUTPUT_HANDLER_TEST_HDRS := \
tensorflow/lite/micro/examples/batt_prediction/output_handler.h \
tensorflow/lite/micro/examples/batt_prediction/constants.h

batt_prediction_SRCS := \
tensorflow/lite/micro/examples/batt_prediction/main.cc \
tensorflow/lite/micro/examples/batt_prediction/main_functions.cc \
tensorflow/lite/micro/examples/batt_prediction/model.cc \
tensorflow/lite/micro/examples/batt_prediction/output_handler.cc \
tensorflow/lite/micro/examples/batt_prediction/constants.cc

batt_prediction_HDRS := \
tensorflow/lite/micro/examples/batt_prediction/model.h \
tensorflow/lite/micro/examples/batt_prediction/output_handler.h \
tensorflow/lite/micro/examples/batt_prediction/constants.h \
tensorflow/lite/micro/examples/batt_prediction/main_functions.h

#Find any platform - specific rules for this example.
include $(wildcard tensorflow/lite/micro/examples/batt_prediction/*/Makefile.inc)

# Tests loading and running the sine model.
$(eval $(call microlite_test,batt_prediction_test,\
$(batt_prediction_TEST_SRCS),$(batt_prediction_TEST_HDRS)))

# Tests producing an output.
$(eval $(call microlite_test,output_handler_test,\
$(OUTPUT_HANDLER_TEST_SRCS),$(OUTPUT_HANDLER_TEST_HDRS)))

# Builds a standalone binary.
$(eval $(call microlite_test,batt_prediction,\
$(batt_prediction_SRCS),$(batt_prediction_HDRS)))
