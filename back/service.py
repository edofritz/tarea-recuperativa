from flask import Flask, request, jsonify
from torchvision import transforms
from PIL import Image
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def image_loader(image_name):
    """Carga imagen y la devuelve como tensor"""
    image = Image.open(image_name)
    image = transform(image).float()
    image = image.unsqueeze(0)  # Agregar dimensión de lote
    return image


# Cargar el modelo previamente guardado con joblib
model = torch.load("models/modelo_con_sin_mascarilla.pth")


@app.route("/", methods=["GET"])
def home():
    return "Hola Service"


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    imagen = Image.open(file.stream)

    if imagen.mode == "RGBA":
        imagen = imagen.convert("RGB")

    imagen_tensor = transform(imagen).unsqueeze(0)

    model.eval()

    with torch.no_grad():
        output = model(imagen_tensor)
        _, predicted = output.max(1)
        logits = model(imagen_tensor)
        probabilidades = F.softmax(logits, dim=1)
        prob_con_mascarilla, prob_sin_mascarilla = probabilidades[0]
        print(f"Probabilidad con mascarilla: {prob_con_mascarilla:.4f}")
        print(f"Probabilidad sin mascarilla: {prob_sin_mascarilla:.4f}")

    resultado = 1 if predicted.item() == 0 else 0
    print(resultado)
    # Usamos la columna de probabilidades de la clase 1 (la segunda columna)

    # Envía la predicción como respuesta
    return jsonify({"prediction": int(resultado)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
