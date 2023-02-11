import pandas as pd
import numpy as np
import config
import json
import pickle

class Diabetic:
    def __init__(self,g,bp,st,ins,bmi,dpf,age):
        self.glucose = g
        self.bloodpressure = bp
        self.skinthickness = st
        self.insulin = ins
        self.bmi  = bmi
        self.diabeticpfunction = dpf
        self.age = age

    def load_model(self):
        with open(config.MODEL_PATH,"rb") as file:
            self.model = pickle.load(file)
        with open(config.PROJECT_DATA_PATH,"r") as file:
            self.project_data = json.load(file)
        with open(config.SCALER_PATH,"rb") as file:
            self.scaler = pickle.load(file)
    
    def predict_diabetes(self):
        self.load_model()
        array = pd.Series(np.zeros(len(self.project_data["columns"])), index =self.project_data["columns"])
        array["Glucose"] = self.glucose
        array["BloodPressure"] = self.bloodpressure
        array["SkinThickness"] = self.skinthickness
        array["Insulin"] = self.insulin
        array["BMI"] = self.bmi
        array["DiabetesPedigreeFunction"] = self.diabeticpfunction
        array["Age"] = self.age

        array = self.scaler.transform([array])

        prediction = self.model.predict(array)[0]

        if prediction == 1:
            prediction = "Person is Diabetic"
        elif prediction == 0:
            prediction = "Person is NOT DIABETIC"

        return prediction
    
if __name__ == "__main__":

    glucose = 120
    bloodpressure = 80
    skinthickness = 15
    insulin = 30
    bmi = 30
    diabeticpfunction = 0.4
    age = 50
    test_diabetes = Diabetic(glucose,bloodpressure,skinthickness,insulin,bmi,diabeticpfunction,age)

    prediction = test_diabetes.predict_diabetes()

    print("************************")
    print(prediction)
    print("************************")
    
