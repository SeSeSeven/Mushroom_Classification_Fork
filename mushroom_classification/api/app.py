from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import os

# Define request and response models
class PredictionRequest(BaseModel):
    input_data: list  # Replace with the appropriate type for your input data

class PredictionResponse(BaseModel):
    prediction: list  # Replace with the appropriate type for your prediction result

# Initialize the FastAPI app
app = FastAPI()

# Load the model
bucket_name = os.getenv('GCS_BUCKET', 'mushrooms_for_mount')
model_path = f"/gcs/{bucket_name}/models/trained_model.pt"

# Ensure the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found.")

# Load the model
model = torch.load(model_path)
model.eval()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Preprocess input data
        input_data = torch.tensor(request.input_data)  # Adjust preprocessing as needed

        # Perform inference
        with torch.no_grad():
            output = model(input_data)
        
        # Postprocess output data
        prediction = output.numpy().tolist()  # Adjust postprocessing as needed

        return PredictionResponse(prediction=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Mushroom Classification API"}

# Define the main entry point for running the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
