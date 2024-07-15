from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision import transforms
import io
import os

vertex_ai = os.getenv("VERTEX_AI", "false").lower() == "true"
model_path = "/gcs/mushroom_test_bucket/models/resnet50.pt" if vertex_ai else "models/resnet50.pt"

model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((288, 288)),
    transforms.ToTensor(),
])

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = transform(image).unsqueeze(0)  

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities, 1)

    return {"class": predicted.item(), "probabilities": probabilities.tolist()}
