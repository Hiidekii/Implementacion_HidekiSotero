import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt

import sys
import os
# Añadir la carpeta raíz del proyecto al path para importar otras carpetas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from approach.ResEmoteNet import ResEmoteNet
from approach.get_dataset import Four4All

# Activar GPU si hay
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Preprocesamiento
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Carga de los datasets (train, validation, test) deben tener las tres carpetas
train_dataset = Four4All(csv_file='../datasets/fer2013_out/train_labels.csv',
                         img_dir='../datasets/fer2013_out/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
train_image, train_label = next(iter(train_loader))

val_dataset = Four4All(csv_file='../datasets/fer2013_out/validation_labels.csv', 
                       img_dir='../datasets/fer2013_out/validation/', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
val_image, val_label = next(iter(val_loader))

test_dataset = Four4All(csv_file='../datasets/fer2013_out/test_labels.csv', 
                        img_dir='../datasets/fer2013_out/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
test_image, test_label = next(iter(test_loader))

# Mostrar forma de los datos de ejemplo
print(f"Train batch: Image shape {train_image.shape}, Label shape {train_label.shape}")
print(f"Validation batch: Image shape {val_image.shape}, Label shape {val_label.shape}")
print(f"Test batch: Image shape {test_image.shape}, Label shape {test_label.shape}")

# Iniciar el modelo
model = ResEmoteNet().to(device)

# Mostrar total de parámetros del modelo
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')

# Definición de función de pérdida y optimizador
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

# Configuración de early stopping
patience = 15
best_val_acc = 0
patience_counter = 0
epoch_counter = 0

# Número de épocas
# num_epochs = 1 # Para probar 
num_epochs = 80

# Listas para guardar métricas por época
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
test_losses = []
test_accuracies = []

# Bucle principal de entrenamiento
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Entrenamiento por lotes
    for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Evaluación sobre el conjunto de test
    model.eval()
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_loss = test_running_loss / len(test_loader)
    test_acc = test_correct / test_total
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    # Evaluación sobre el conjunto de validación
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_running_loss / len(val_loader)
    val_acc = val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    # Mostrar métricas por epoch
    print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Train Accuracy: {train_acc}, "
          f"Test Loss: {test_loss}, Test Accuracy: {test_acc}, "
          f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

    epoch_counter += 1

    # Guardar modelo más pro
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0 
        torch.save(model.state_dict(), '../models/fer2013_model.pth')
    else:
        patience_counter += 1
        print(f"No improvement in validation accuracy for {patience_counter} epochs.")
    
    # Criterio early stopping
    if patience_counter > patience:
        print("Stopping early due to lack of improvement in validation accuracy.")
        break

# Guardar las métricas en csv
df = pd.DataFrame({
    'Epoch': range(1, epoch_counter+1),
    'Train Loss': train_losses,
    'Test Loss': test_losses,
    'Validation Loss': val_losses,
    'Train Accuracy': train_accuracies,
    'Test Accuracy': test_accuracies,
    'Validation Accuracy': val_accuracies
})
df.to_csv('result_four4all_80epch.csv', index=False)
