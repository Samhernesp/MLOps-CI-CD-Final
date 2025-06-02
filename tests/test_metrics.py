import pytest
import pandas as pd
import numpy as np
import onnxruntime as ort
from sklearn.metrics import mean_absolute_error
import os

MODEL_PATH = "local_model/predictor_model.onnx"
VALIDATION_DATA_PATH = "data/Salary_dataset.csv"

MAE_THRESHOLD = 5000.0 

@pytest.fixture(scope="session")
def onnx_session():
    """Carga el modelo ONNX una sola vez para todas las pruebas en este archivo."""
    if not os.path.exists(MODEL_PATH):
        pytest.fail(f"Model file not found at: {MODEL_PATH}")
    return ort.InferenceSession(MODEL_PATH)

def test_model_mae_performance(onnx_session):
    """
    Verifica que el Error Absoluto Medio (MAE) del modelo en un set de
    validación esté por debajo del umbral definido.
    """
    # 1. Cargar los datos de validación
    try:
        validation_data = pd.read_csv(VALIDATION_DATA_PATH)
    except FileNotFoundError:
        pytest.fail(f"Validation data not found at: {VALIDATION_DATA_PATH}")

    X_val = validation_data[['YearsExperience']].values.astype(np.float32)
    y_true = validation_data['Salary'].values

    # 2. Realizar predicciones para todo el set de datos
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    
    predictions = onnx_session.run([output_name], {input_name: X_val})[0]
    # La salida puede ser un array de arrays, lo aplanamos
    y_pred = predictions.flatten()

    # 3. Calcular la métrica (Error Absoluto Medio)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n[INFO] Calculated Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"[INFO] MAE Threshold: ${MAE_THRESHOLD:,.2f}")

    # 4. Verificar que la métrica esté dentro del umbral
    assert mae < MAE_THRESHOLD, (
        f"Model MAE ${mae:,.2f} is higher than the threshold of ${MAE_THRESHOLD:,.2f}. "
        "Model performance has degraded."
    )