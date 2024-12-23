import numpy as np
import tflite_runtime.interpreter as tflite

def tflite_load(path):
    # Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()

    return interpreter

def tflite_predict(interpreter, data):
    # Test the model on random input data.
    input_data = np.array(data, dtype=np.float32)
    # print(input_data.shape)
    # input_data = input_data[np.newaxis, ...]
    # input_data = np.expand_dims(input_data, axis=0)
    input_data = np.expand_dims(input_data, axis=2)
    # print(input_data.shape)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    return output_data
