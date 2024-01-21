from flask import Flask, request, jsonify
from torchvision import transforms
from PIL import Image
from flask_cors import CORS
import tritonclient.http as httpclient
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

triton_client = httpclient.InferenceServerClient(url="triton:8000")

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@app.route("/", methods=["GET"])
def home():
    return "Hola Service"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]

        imagen = Image.open(file.stream)

        if imagen.mode == "RGBA":
            imagen = imagen.convert("RGB")

        imagen_tensor = transform(imagen).unsqueeze(0).numpy()

        # Crear la solicitud de inferencia
        inputs = [httpclient.InferInput("input", imagen_tensor.shape, "FP32")]
        inputs[0].set_data_from_numpy(imagen_tensor)

        # Realizar la solicitud de inferencia
        outputs = [httpclient.InferRequestedOutput("output")]
        response = triton_client.infer(
            model_name="modelo_con_sin_mascarilla", inputs=inputs, outputs=outputs
        )

        # Procesar la respuesta
        output_data = response.as_numpy("output")
        predicted = np.argmax(output_data, axis=1)
        resultado = int(predicted[0])

        return jsonify({"prediction": resultado})
    except Exception as e:
        error_message = str(e)
        return jsonify({"error": error_message}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
