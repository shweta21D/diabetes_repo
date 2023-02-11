from flask import Flask, jsonify, request
import config
from utilities import Diabetic

app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to Diabetic Prediction App"

@app.route("/predict", methods = ["Post"])
def get_prediction():
    data = request.form
    glucose = eval(data["g"])
    bloodpressure = eval(data["bp"])
    skintickness = eval(data["st"])
    insulin = eval(data["ins"])
    bmi = eval(data["bmi"])
    diabeticpfunction = eval(data["dpf"])
    age = eval(data["age"])

    diabetes_test = Diabetic(glucose,bloodpressure,skintickness,insulin,bmi,diabeticpfunction,age)
    prediction = diabetes_test.predict_diabetes()
    print("*************************")
    print(prediction)
    print("*************************")

    return jsonify({"Result":prediction})

if __name__ == "__main__":
    app.run(debug = True, port = config.PORT_NUMBER, host = config.HOST)
