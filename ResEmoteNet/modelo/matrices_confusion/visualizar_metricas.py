import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from approach.ResEmoteNet import ResEmoteNet
from approach.get_dataset import Four4All

class_map = {
    0: 'happy',
    1: 'surprise',
    2: 'sad',
    3: 'angry',
    4: 'disgust',
    5: 'fear',
    6: 'neutral'
}

# Etiquetas del dataset FER2013
emotion_labels = [class_map[i] for i in range(7)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocesamiento
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Dataset de test
test_dataset = Four4All(csv_file='../datasets/rafdb_out/test_labels.csv',
                        img_dir='../datasets/rafdb_out/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Modelo
model = ResEmoteNet().to(device)
checkpoint = torch.load('../models/rafdb_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Matriz de confusión
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=emotion_labels, yticklabels=emotion_labels, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Matriz de Confusión - RAFDB")
plt.show()

# Métricas detalladas
report = classification_report(all_labels, all_preds, target_names=emotion_labels, digits=4)
print("Reporte de Clasificación:\n")
print(report)