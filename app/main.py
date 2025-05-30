import onnxruntime as ort
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import os


class ModelInput(BaseModel):
    experience_years: float

app = FastAPI(title="ML Model Server")

MODEL_PATH = "local_model/predictor_model.onnx"
LOG_FILE_PATH = "local_predictions.log"

session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name


@app.get("/")
def read_root():
    return {"status": "ok", "message": "ML Model Server is running."}


@app.post("/predict")
def predict(data: ModelInput):
    """
    Realiza una predicción usando el modelo ONNX cargado.
    """
   
    input_data = np.array(data.experience_years, dtype=np.float32).reshape(1, 1)

    result = session.run(None, {input_name: input_data})


    salary_predicted = float(result[0][0][0])

    # print(f"Predicted Salary: {salary_predicted}")

    log_entry = f"{datetime.now().isoformat()} - Years of Experience: {data.experience_years} - Salary Prediction: {salary_predicted} \n"
    with open(LOG_FILE_PATH, "a") as log_file:
        log_file.write(log_entry)

    return {
        "experience_years": data.experience_years,
        "prediction": salary_predicted
    }