import os
import pandas as pd

# Ruta donde están las imágenes ya renombradas del conjunto que quieras procesar (por ejemplo validation)
path = '../datasets/rafdb_out/validation'

# Diccionario para mapear el nombre de la emoción al número de clase
label_mapping = {
    "happy": 0,
    "surprise": 1,
    "sad": 2,
    "angry": 3,
    "disgust": 4,
    "fear": 5,
    "neutral": 6
}

image_data = []

# Recorremos todas las imágenes en la carpeta
# Asumimos que los nombres son del tipo: val_123_angry.jpg
for filename in os.listdir(path):
    if filename.endswith(".jpg") or filename.endswith(".png"):  
        # Sacamos el nombre de la emoción del archivo
        label_name = filename.split('_')[-1].split('.')[0]
        label_value = label_mapping.get(label_name)  # Lo convertimos al número de clase

        # Si la emoción está en el diccionario, la guardamos
        if label_value is not None:  
            image_data.append([filename, label_value])

# Convertimos la lista a un DataFrame
df = pd.DataFrame(image_data, columns=["ImageName", "Label"])

# Guardamos el CSV en la ruta deseada
csv_file_path = '../datasets/rafdb_out/validation_labels.csv'
df.to_csv(csv_file_path, index=False, header=False)

print(f"CSV creado en: {csv_file_path}")