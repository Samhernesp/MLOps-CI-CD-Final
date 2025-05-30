# tests/test_api.py
from fastapi.testclient import TestClient 
from app.main import app, LOG_FILE_PATH 
import os
import pytest

client = TestClient(app)

@pytest.fixture(autouse=True)
def cleanup_log_file():
    """Limpia el archivo de log antes y después de cada prueba."""
    if os.path.exists(LOG_FILE_PATH):
        os.remove(LOG_FILE_PATH)
    yield
    if os.path.exists(LOG_FILE_PATH):
        os.remove(LOG_FILE_PATH)


def test_read_root():
    """Prueba que el endpoint raíz funciona."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Salary Predictor ML Model Server is running."

def test_predict_endpoint_valid_input():
    """Prueba el endpoint de predicción con una entrada válida."""
    experience_years = 5.0
    response = client.post(
        "/predict",
        json={"experience_years": experience_years}
    )
    assert response.status_code == 200
    response_json = response.json()
    assert "experience_years" in response_json
    assert "predicted_salary" in response_json
    assert response_json["experience_years"] == experience_years
    assert isinstance(response_json["predicted_salary"], float)

    # Verificar que se creó el log
    assert os.path.exists(LOG_FILE_PATH)
    with open(LOG_FILE_PATH, "r") as f:
        log_content = f.read()
        assert f"Years of Experience: {experience_years}" in log_content
        assert "Salary Prediction:" in log_content


def test_predict_endpoint_invalid_input_type():
    """Prueba el endpoint de predicción con un tipo de entrada inválido."""
    response = client.post(
        "/predict",
        json={"experience_years": "cinco"} # String en lugar de float
    )
    assert response.status_code == 422 # Error de validación de Pydantic


def test_get_logs_empty():
    """Prueba el endpoint /logs cuando no hay logs."""
    # El fixture cleanup_log_file ya asegura que el archivo no existe o está vacío
    response = client.get("/logs")
    assert response.status_code == 200
    # Podría ser "Log file not found." o simplemente una lista vacía si el archivo se crea vacío
    # Depende de si el archivo se crea al inicio o solo al escribir el primer log.
    # Con la implementación actual, si no hay predicciones, el archivo no existe.
    assert response.json().get("logs") == [] or "Log file not found" in response.json().get("message", "")


def test_get_logs_with_content():
    """Prueba el endpoint /logs después de realizar algunas predicciones."""
    # Realizar algunas predicciones para generar logs
    client.post("/predict", json={"experience_years": 2.0})
    client.post("/predict", json={"experience_years": 7.5})

    response = client.get("/logs")
    assert response.status_code == 200
    response_json = response.json()
    assert "logs" in response_json
    assert len(response_json["logs"]) == 2
    assert "Years of Experience: 2.0" in response_json["logs"][0]
    assert "Years of Experience: 7.5" in response_json["logs"][1]

# Considera una prueba para una métrica o un valor conocido si tienes uno.
# Por ejemplo, si sabes que para 5 años de experiencia el salario debería ser X:
# def test_prediction_known_value():
#     experience_years = 5.0
#     # Supongamos que tu modelo ONNX predice ~75000 para 5 años
#     # Esto depende de tu modelo específico.
#     # Primero ejecuta la predicción, observa el resultado, y luego fíjalo aquí.
#     expected_salary_around = 75000.0 
#     tolerance = 5000.0 # Tolerancia para la predicción

#     response = client.post(
#         "/predict",
#         json={"experience_years": experience_years}
#     )
#     assert response.status_code == 200
#     predicted_salary = response.json()["predicted_salary"]
#     assert expected_salary_around - tolerance <= predicted_salary <= expected_salary_around + tolerance