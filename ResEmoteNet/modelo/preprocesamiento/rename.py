import os
import shutil

# Ruta donde están las carpetas originales (train, test, validation)
source_dir = '../datasets/rafdb_in'

# Ruta donde se guardarán las imágenes con nombres nuevos
destination_dir = '../datasets/rafdb_out'

# Diccionario para renombrar los prefijos según el tipo de conjunto
folder_mapping = {
    'test': 'test', 
    'train': 'train', 
    'validation': 'val'
}

# Recorremos cada carpeta (train, test, validation)
for folder in ['test', 'train', 'validation']:
    folder_path = os.path.join(source_dir, folder)
    
    # Recorremos las subcarpetas por clase (happy, sad, etc.)
    for class_folder in os.listdir(folder_path):
        class_folder_path = os.path.join(folder_path, class_folder)
        
        # Recorremos cada imagen dentro de esa clase
        for index, image in enumerate(os.listdir(class_folder_path)):
            # Sacamos el nombre y extensión del archivo
            image_name, image_ext = os.path.splitext(image)
            
            # Armamos el nuevo nombre con formato: tipo_index_clase.jpg
            new_image_name = f"{folder_mapping[folder]}_{index}_{class_folder}{image_ext}"
            
            # Movemos y renombramos la imagen a la carpeta destino
            shutil.move(
                os.path.join(class_folder_path, image),
                os.path.join(destination_dir, folder, new_image_name)
            )