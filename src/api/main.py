from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import PredictionRequest, PredictionResponse
import pandas as pd
import mlflow.pyfunc
from src.predict import preprocess_input

app = FastAPI(title="Credit Risk Model API")

# Load model from MLflow Registry
model = mlflow.pyfunc.load_model(model_uri="models:/RandomForest/1")  # version 1


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: PredictionRequest):
    try:
        # Convert to DataFrame
        df = pd.read_csv("data/processed/data_with_target.csv")

        X = df.drop(columns=["CustomerId", "is_high_risk", "TransactionStartTime"])
        df = pd.DataFrame([input_data.dict()])
        print("Raw input shape:", df.shape)

        # Preprocess to expected format
        processed = preprocess_input(df)
        print("Processed shape:", processed.shape)

        # Predict
        prediction = model.predict(processed[X.columns])
        return PredictionResponse(risk_probability=float(prediction[0]))

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{e}")