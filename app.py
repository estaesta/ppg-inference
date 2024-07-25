from flask import Flask, render_template, request, jsonify
from extraction_function import preprocess_all
from tensorflow.keras.models import load_model
import time
import numpy as np
import joblib

app = Flask(__name__)

label_dict = {'baseline': 1, 'stress': 2, 'amusement': 0}
int_to_label = {1: 'baseline', 2: 'stress', 0: 'amusement'}

#model global variable
model = None

@app.route("/", methods=["GET"])
def index():
    return render_template("VerityCollectionToolv0.051.html")

@app.route("/index", methods=["GET"])
def index2():
    return render_template("index.html")

#change model
@app.route("/change_model", methods=["POST"])
def change_model():
    global model
    # model: svm, lstm, lstmfcn, cnn
    # let response = await fetch("/change_model", {
    #   method: "POST",
    #   headers: {
    #     "Content-Type": "application/json",
    #   },
    #   body: JSON.stringify({ model: model }),
    # });
    modelName = request.get_json(force=True)["model"]


    match modelName:
        case "svm":
            model = joblib.load('./model/svm_tri_1swin.pkl')
        case "lstmfcn":
            model = load_model('./model/lstmfcn-tri-1swin-256bs-seed0.h5')
        case "cnn":
            model = load_model('./model/cnn-tri-1swin-256bs-seed0.h5')
        case _:
            model = None
    return jsonify({"model": modelName})


@app.route("/predict", methods=["POST"])
def predict():
    start = time.time()
    global model
    # Get the data from the POST request.
    data = request.get_json(force=True)
    # print(data)
    ppg_signal = data["ppg_signal"]

    preprocessed_ppg, _, bp_bvp, hr = preprocess_all(ppg_signal)
    # print(f'Preprocessed PPG: {preprocessed_ppg}')
    # print(preprocessed_ppg)

    # Make prediction and return result.
    # cnn
    # make the model load first time only
    # model = load_model('./model/cnn-tri-256bs-seed0.h5')
    if model is None:
        # model = load_model('./model/cnn-tri-256bs-seed0.h5')
        # model = load_model('./model/lstmfcn-tri-1swin-256bs-seed0.h5')
        model = load_model('./model/cnn-tri-1swin-256bs-seed0.h5')
    result = model.predict(preprocessed_ppg)
    # one-hot encoding to class
    result = np.argmax(result, axis=1)
    result = int_to_label[result[0]]
    # svm
    # model = joblib.load('./model/svm_tri.pkl')
    # # model = joblib.load('./model/svm_tri_1swin.pkl')
    # result = model.predict(preprocessed_ppg)
    # print(result)
    # result = int_to_label[result[0]]
    # return bp_bvp too
    # bp_bvp = bp_bvp.tolist()
    # return hr too
    # hr = hr.tolist()
    result = {"result": result, "hr": hr[-1:], "mean_hr": np.mean(hr)}
    # return jsonify({"result": result, "bp_bvp": bp_bvp})
    result = jsonify(result)
    print(f'Time taken: {time.time() - start}')
    return result

@app.route("/test", methods=["GET"])
def test():
    return "Hello, World!"

if __name__ == "__main__":
    app.run(port=5005)
