from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from extraction_function import preprocess_all

# from utils import *
import time
import numpy as np
import joblib
import resource
from threading import Lock

app = Flask(__name__)
# app.config["SECRET_KEY"] = "secret!"
# socketio = SocketIO(app)

label_dict = {"baseline": 1, "stress": 2, "amusement": 0}
int_to_label = {1: "baseline", 2: "stress", 0: "amusement"}

# model global variable
model = None
model_lock = Lock()
model_name = ""

tf_support = False
tfl_support = True

if tf_support:
    from tensorflow.keras.models import load_model
if tfl_support:
    import tflite_runtime.interpreter as tflite


def tflite_load(path):
    interpreter = tflite.Interpreter(model_path=path)
    interpreter.allocate_tensors()

    return interpreter

model = tflite_load("./model/tflite/bilstm-tri-1swin-256bs-seed0.tflite")
model_name = "bilstm_tfl"

def tflite_predict(interpreter, data):
    input_data = np.array(data, dtype=np.float32)
    input_data = np.expand_dims(input_data, axis=2)
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    return output_data

# @app.before_request
# def before_request():
#     g.request_start_time = time.time()
#
# @app.after_request
# def after_request(response):
#     # if response is json, save the time to csv
#     request_time = (time.time() - g.request_start_time) * 1000
#     with open("csv_output/request.csv", "a") as f:
#         f.write(f"{request_time}\n")
#         f.close()
#     return response


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")
    # return render_template("VerityCollectionToolv0.051.html")


@app.route("/ppg", methods=["GET"])
def ppg():
    return render_template("ppg.html")


# change model
@app.route("/change_model", methods=["POST"])
def change_model():
    global model
    global model_name
    request_model = request.get_json(force=True)["model"]

    if request_model != model_name:
        with model_lock:
            if request_model != model_name:
                match request_model:
                    case "svm":
                        model = joblib.load("./model/svm_tri_1swin.pkl")
                    case "lstm512":
                        model = load_model("./model/lstm512-tri-1swin-256bs-seed0.h5")
                    case "lstm512_tfl":
                        model = tflite_load(
                            "./model/tflite/lstm512-tri-1swin-256bs-seed0.tflite"
                        )
                    case "lstm256":
                        model = load_model("./model/lstm256-tri-1swin-256bs-seed0.h5")
                    case "lstm256_tfl":
                        model = tflite_load(
                            "./model/tflite/lstm256-tri-1swin-256bs-seed0.tflite"
                        )
                    case "bilstm":
                        model = load_model("./model/bilstm-tri-1swin-256bs-seed0.h5")
                    case "bilstm_tfl":
                        model = tflite_load(
                            "./model/tflite/bilstm-tri-1swin-256bs-seed0.tflite"
                        )
                    case "lstmfcn":
                        model = load_model("./model/lstmfcn-tri-1swin-256bs-seed0.h5")
                    case "lstmfcn_tfl":
                        model = tflite_load(
                            "./model/tflite/lstmfcn-tri-1swin-256bs-seed0.tflite"
                        )
                    case "cnn":
                        model = load_model("./model/cnn-tri-1swin-256bs-seed0.h5")
                    case "cnn_tfl":
                        model = tflite_load("./model/tflite/cnn-tri-1swin-256bs-seed0.tflite")
                    case "none":
                        model = None
                    case _:
                        model = None
                model_name = request_model
        print(f"Model changed to {model_name}")
    return jsonify({"model": request_model})


@app.route("/predict", methods=["POST"])
def predict():
    start = time.time()
    global model
    global model_name
    # Get the data from the POST request.
    data = request.get_json(force=True)
    # print(data)
    ppg_signal = data["ppg_signal"]
    name = data["name"]
    # request_model = data["model"]
    # # check if model is the same. If not, load the new model
    # if request_model != model_name:
    #     with model_lock:
    #         if request_model != model_name:
    #             match request_model:
    #                 case "svm":
    #                     model = joblib.load("./model/svm_tri_1swin.pkl")
    #                 case "lstm512":
    #                     model = load_model("./model/lstm512-tri-1swin-256bs-seed0.h5")
    #                 case "lstm512_tfl":
    #                     model = tflite_load(
    #                         "./model/tflite/lstm512-tri-1swin-256bs-seed0.tflite"
    #                     )
    #                 case "lstm256":
    #                     model = load_model("./model/lstm256-tri-1swin-256bs-seed0.h5")
    #                 case "lstm256_tfl":
    #                     model = tflite_load(
    #                         "./model/tflite/lstm256-tri-1swin-256bs-seed0.tflite"
    #                     )
    #                 case "bilstm":
    #                     model = load_model("./model/bilstm-tri-1swin-256bs-seed0.h5")
    #                 case "bilstm_tfl":
    #                     model = tflite_load(
    #                         "./model/tflite/bilstm-tri-1swin-256bs-seed0.tflite"
    #                     )
    #                 case "lstmfcn":
    #                     model = load_model("./model/lstmfcn-tri-1swin-256bs-seed0.h5")
    #                 case "lstmfcn_tfl":
    #                     model = tflite_load(
    #                         "./model/tflite/lstmfcn-tri-1swin-256bs-seed0.tflite"
    #                     )
    #                 case "cnn":
    #                     model = load_model("./model/cnn-tri-1swin-256bs-seed0.h5")
    #                 case "cnn_tfl":
    #                     model = tflite_load(
    #                         "./model/tflite/cnn-tri-1swin-256bs-seed0.tflite"
    #                     )
    #                 case _:
    #                     model = None
    #             model_name = request_model
    #     print(f"Model changed to {model_name}")
    if model is None:
        return jsonify({"result": "No model loaded"})

    preprocessed_ppg, _, _, hr = preprocess_all(ppg_signal)

    # if model is None:
    #     model = load_model('./model/cnn-tri-1swin-256bs-seed0.h5')
    #     model_name = "cnn"

    # predict
    # check if tflite
    if model_name[-3:] == "tfl":
        result = tflite_predict(model, preprocessed_ppg)
        # print(result)
    elif model_name[-2:] == "tf":
        result = model.predict(preprocessed_ppg, verbose=0)
    else:
        result = model.predict(preprocessed_ppg)
    # process memory usage in MB
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"Memory usage: {mem} MB")

    # one-hot encoding to class
    if model_name != "svm":
        result = np.argmax(result, axis=1)
    result = int_to_label[result[0]]
    output = {"result": result, "hr": hr[-1:], "mean_hr": np.mean(hr)}
    output = jsonify(output)

    # time in ms
    time_taken = (time.time() - start) * 1000
    print(f"Time taken: {time_taken} ms")

    # write to csv
    with open("csv_output/results.csv", "a") as f:
        f.write(f"{model_name}, {result}, {time_taken}, {mem}\n")
        f.close()

    # send to monitor
    # send_to_monitor(output)
    # change: send the name too
    print(name)
    print(type(name))
    # send_to_monitor({"stress": result, "name": name, "hr": hr[-1:]})
    return output


# @app.route("/test", methods=["GET"])
# def test():
#     # send to monitor every 1 second infinitely
#     name1 = "test1"
#     name2 = "test2"
#     while True:
#         #random result
#         stress1 = np.random.choice(["baseline", "stress", "amusement"])
#         hr1 = np.random.randint(60, 100)
#         stress2 = np.random.choice(["baseline", "stress", "amusement"])
#         hr2 = np.random.randint(60, 100)
#         send_to_monitor({"stress": stress1, "name": name1, "hr": hr1})
#         send_to_monitor({"stress": stress2, "name": name2, "hr": hr2})
#         time.sleep(1)
#     return "Hello, World!"

# @app.route("/monitor", methods=["GET"])
# def monitor():
#     return render_template("monitor.html")
#
# ### SocketIO for real-time monitoring
# @socketio.on("connect")
# def test_connect():
#     print("Client connected")
#
# @socketio.on("disconnect")
# def test_disconnect():
#     print("Client disconnected")
#
# # send prediction result to monitor
# def send_to_monitor(result):
#     socketio.emit("monitor", result)
#     print("Sent to monitor")


if __name__ == "__main__":
    app.run(ssl_context="adhoc")
