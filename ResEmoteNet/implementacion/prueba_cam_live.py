import cv2
import cv2.data
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time

import sys
import os
# Agregamos la carpeta del proyecto al path para poder usar otras carpetas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modelo.approach.ResEmoteNet import ResEmoteNet

# Seleccionamos el dispositivo: GPU si hay, sino CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Lista de emociones en el orden de salida del modelo
emotions = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']

# Cargamos el modelo y sus pesos entrenados
model = ResEmoteNet().to(device)
checkpoint = torch.load('../modelo/models/fer2013_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Preprocesameinto
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Cargamos el detector de rostros de OpenCV (Haar cascada), reemplazar con YOLO o RetinaFace despues de optimizar
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Iniciamos la cámara (por defecto: webcam) --> a futuro se captura la pantalla
video_capture = cv2.VideoCapture(0)

# Texto que se muestra sobre la imagen
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.2
font_color = (0, 255, 0)  # Verde
thickness = 3
line_type = cv2.LINE_AA

# Inciar variable
max_emotion = ''


# Función pa obtener los scores de emociones a partir de un frame (1 rostro)
def detect_emotion(video_frame):
    vid_fr_tensor = transform(video_frame).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(vid_fr_tensor)
        probabilities = F.softmax(outputs, dim=1)
    scores = probabilities.cpu().numpy().flatten()
    rounded_scores = [round(score, 2) for score in scores]
    return rounded_scores

# Detecta emoción dominante
def get_max_emotion(x, y, w, h, video_frame):
    crop_img = video_frame[y : y + h, x : x + w]
    pil_crop_img = Image.fromarray(crop_img)
    rounded_scores = detect_emotion(pil_crop_img)    
    max_index = np.argmax(rounded_scores)
    max_emotion = emotions[max_index]
    return max_emotion

# Muestra emoción dominante sobre el rostro
def print_max_emotion(x, y, video_frame, max_emotion):
    org = (x, y - 15)
    cv2.putText(video_frame, max_emotion, org, font, font_scale, font_color, thickness, line_type)

# Muestra todas las emociones con su score
def print_all_emotion(x, y, w, h, video_frame):
    crop_img = video_frame[y : y + h, x : x + w]
    pil_crop_img = Image.fromarray(crop_img)
    rounded_scores = detect_emotion(pil_crop_img)
    org = (x + w + 10, y - 20)
    for index, value in enumerate(emotions):
        emotion_str = (f'{value}: {rounded_scores[index]:.2f}')
        y = org[1] + 40
        org = (org[0], y)
        cv2.putText(video_frame, emotion_str, org, font, font_scale, font_color, thickness, line_type)

# Detección de rostros y visualización de resultados
def detect_bounding_box(video_frame, counter):
    global max_emotion
    gray_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    
    for (x, y, w, h) in faces:
        # Dibujamos el recuadro del rostro
        cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Cada N frames, actualizamos la emoción dominante
        if counter == 0:
            max_emotion = get_max_emotion(x, y, w, h, video_frame) 
        
        # Mostramos la emoción dominante
        print_max_emotion(x, y, video_frame, max_emotion)
        
        # Mostramos los scores de todas las emociones
        print_all_emotion(x, y, w, h, video_frame)

    return faces

# Contador para ejecutar la predicción completa cada N frames
counter = 0
evaluation_frequency = 5

# Loop principal en tiempo real
prev_time = time.time()

while True:
    current_time = time.time()
    delta_time = current_time - prev_time
    fps = 1 / delta_time if delta_time > 0 else 0
    prev_time = current_time

    # Leemos un frame del video
    result, video_frame = video_capture.read()
    if result is False:
        break 

    # Detectamos y procesamos los rostros
    faces = detect_bounding_box(video_frame, counter)

    # Mostrar FPS pa rendimiento
    cv2.putText(
        video_frame,
        f'FPS: {fps:.2f}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Mostramos el frame con resultados
    cv2.imshow("ResEmoteNet", video_frame)

    # También imprimimos los FPS en consola
    print(f"FPS: {fps:.2f}")

    # Salimos del loop si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Incrementamos el contador y lo reseteamos cada N frames
    counter += 1
    if counter == evaluation_frequency:
        counter = 0

# Liberamos la cámara y cerramos la ventana
video_capture.release()
cv2.destroyAllWindows()
