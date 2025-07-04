import os
import random
import shutil

# Ruta a tu directorio de entrenamiento
train_dir = "../datasets/ferYrafdbYaffect/train"  # Cambia esto si usas otro dataset

# Umbral máximo de imágenes por clase
max_images = 2500

# Crear una carpeta nueva para almacenar las clases balanceadas
balanced_dir = train_dir + "_balanced"
os.makedirs(balanced_dir, exist_ok=True)

# Procesar cada carpeta/emoción
for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    if os.path.isdir(class_path):
        images = os.listdir(class_path)
        print(f"[{class_name}] Total imágenes: {len(images)}")

        # Mezclar aleatoriamente las imágenes
        random.shuffle(images)

        # Seleccionar hasta el límite definido
        selected_images = images[:max_images]

        # Crear carpeta destino
        dest_class_path = os.path.join(balanced_dir, class_name)
        os.makedirs(dest_class_path, exist_ok=True)

        # Copiar las imágenes seleccionadas
        for img_name in selected_images:
            src_img_path = os.path.join(class_path, img_name)
            dst_img_path = os.path.join(dest_class_path, img_name)
            shutil.copy2(src_img_path, dst_img_path)

        print(f"   → Copiadas {len(selected_images)} imágenes a {dest_class_path}")

print("\n✅ Balanceo completo. Usa la nueva carpeta:", balanced_dir)
