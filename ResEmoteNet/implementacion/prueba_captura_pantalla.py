import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time
import sys
import os
from datetime import datetime

# Agregamos la carpeta del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modelo.approach.ResEmoteNet import ResEmoteNet

# Dispositivo: GPU si hay, sino CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Emociones
emotions = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']

# Modelo
model = ResEmoteNet().to(device)
checkpoint = torch.load('../modelo/models/fer2013_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Transformación de imagen
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Detector de rostros (Haar cascade)
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Fuente para mostrar texto
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.2
font_color = (0, 255, 0)
thickness = 3
line_type = cv2.LINE_AA

# Historial de emociones detectadas
last_emotions = []

# Batch processing

def detect_emotion_batch(pil_images):
    tensors = [transform(img) for img in pil_images]
    batch_tensor = torch.stack(tensors).to(device)
    with torch.no_grad():
        outputs = model(batch_tensor)
        probabilities = F.softmax(outputs, dim=1).cpu().numpy()
    return probabilities

# Detección y visualización de emociones

def detect_bounding_box(video_frame, counter):
    global last_emotions

    gray_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))

    crops = []
    coords = []

    for (x, y, w, h) in faces:
        crop_img = video_frame[y:y+h, x:x+w]
        pil_img = Image.fromarray(crop_img)
        crops.append(pil_img)
        coords.append((x, y, w, h))

    if counter == 0 and crops:
        scores_batch = detect_emotion_batch(crops)
        last_emotions = scores_batch

    for i, (x, y, w, h) in enumerate(coords):
        cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if i < len(last_emotions):
            scores = last_emotions[i]
            max_index = np.argmax(scores)
            max_emotion = emotions[max_index]

            label = f"Rostro {i+1} - {max_emotion.capitalize()}"
            org = (x, y - 10)
            cv2.putText(video_frame, label, org, font, font_scale, font_color, thickness, line_type)

            if counter == 0:
                timestamp = datetime.now().strftime("%I:%M%p").lstrip("0").lower()
                print(f"Rostro {i+1}, {max_emotion.capitalize()}, {timestamp}")

    return faces

# Abrir cámara virtual de OBS (ajustar el índice si es necesario)
video_capture = cv2.VideoCapture(1)

# Bucle principal
evaluation_frequency = 5
counter = 0
prev_time = time.time()

while True:
    current_time = time.time()
    delta_time = current_time - prev_time
    fps = 1 / delta_time if delta_time > 0 else 0
    prev_time = current_time

    result, video_frame = video_capture.read()
    if not result:
        continue

    faces = detect_bounding_box(video_frame, counter)

    cv2.putText(
        video_frame,
        f'FPS: {fps:.2f}',
        (10, 30),
        font,
        1,
        (0, 255, 255),
        2,
        line_type
    )

    cv2.imshow("ResEmoteNet - OBS Virtual Camera", video_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    counter += 1
    if counter == evaluation_frequency:
        counter = 0

video_capture.release()
cv2.destroyAllWindows()
