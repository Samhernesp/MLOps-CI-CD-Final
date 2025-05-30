import onnxruntime as ort
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import os


class ModelInput(BaseModel):
    experience_years: float

app = FastAPI(title="MLOps Model Server")

MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "local_model/predictor_model.onnx")
LOG_FILE_PATH = os.getenv("LOG_FILE", "local_predictions.log")
LOG_DESTINATION = os.getenv("LOG_DESTINATION", "local")

try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name ###
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    session = None 
    input_name = None
    output_name = None

def log_action(entry: str):
    """Escribe una entrada de log según el destino configurado."""

    if LOG_DESTINATION == "local":

        try:
            with open(LOG_FILE_PATH, "a") as log_file:
                log_file.write(entry + "\n")
        except Exception as e:
            print(f"Error writing to local log file: {e}")

    # elif LOG_DESTINATION == "s3":
    #     # futura_logica_s3(entry)
    #     pass
    else:
        print(f"Log Entry (console): {entry}")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Salary Predictor ML Model Server is running."}


@app.post("/predict")
def predict(data: ModelInput):
    """
    Realiza una predicción usando el modelo ONNX cargado.
    """

    if not session:
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")
    
    try:
        input_data = np.array(data.experience_years, dtype=np.float32).reshape(1, 1)

        result = session.run([output_name], {input_name: input_data})

        salary_predicted = float(result[0][0][0])

        # print(f"Predicted Salary: {salary_predicted}")

        log_entry = f"{datetime.now().isoformat()} - Years of Experience: {data.experience_years} - Salary Prediction: {salary_predicted:.2f}"
        log_action(log_entry)

        return {
            "experience_years": data.experience_years,
            "predicted_salary": round(salary_predicted, 2)
        }
    except Exception as e:
        log_action(f"ERROR in prediction: {e} - Input: {data.experience_years}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.get("/logs")
def get_logs():
    """Retorna el contenido del archivo de logs local."""

    if LOG_DESTINATION != "local":
        return {"message": "Logs are not stored locally for the current configuration."}
    try:

        with open(LOG_FILE_PATH, "r") as log_file:
            logs = log_file.readlines()
        logs_cleaned = [line.strip() for line in logs]
        return {"logs": logs_cleaned}
    
    except FileNotFoundError:
        return {"logs": [], "message": "Log file not found."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading log file: {str(e)}")