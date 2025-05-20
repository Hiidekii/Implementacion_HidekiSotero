import base64
import io
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
import sys, os
from datetime import datetime

# Setup Flask
app = Flask(__name__)
CORS(app)

# Configurar PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Agregar el modelo al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from approach.ResEmoteNet import ResEmoteNet

# Emociones
emotions = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']

# Cargar modelo
model = ResEmoteNet().to(device)
checkpoint = torch.load('../models/fer2013_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Preprocesamiento
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Cargar detector de rostros
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Ruta para detección de emociones
@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    try:
        data = request.get_json()
        img_data = base64.b64decode(data['image'].split(",")[1])
        pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")

        # Convertir a formato OpenCV
        cv_img = np.array(pil_img)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))

        result = []
        timestamp = datetime.now().strftime("%I:%M%p").lstrip("0").lower()  # Ej: 10:42pm

        for i, (x, y, w, h) in enumerate(faces):
            crop = cv_img[y:y+h, x:x+w]
            face_pil = Image.fromarray(crop)
            tensor = transform(face_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(tensor)
                prob = F.softmax(output, dim=1)
                emotion_idx = torch.argmax(prob).item()
                emotion_label = emotions[emotion_idx].capitalize()

            result.append({
                "rostro": f"Rostro {i + 1}",
                "emocion": emotion_label,
                "hora": timestamp,
                "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            })

        return jsonify({"result": result})

    except Exception as e:
        print("❌ Error al procesar la imagen:", e)
        return jsonify({"result": [], "error": str(e)}), 500

# Iniciar servidor
if __name__ == '__main__':
    app.run(debug=True)
