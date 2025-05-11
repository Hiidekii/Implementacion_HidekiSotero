Intalación de dependencias:
- Pueden existir dificultades para añadir la dependencia dlib asi que de preferencia instalar del wheel precargado
- Las dependencias necesarias para la ejecución del codigo se encuentran en el archivo requirements.txt
- Se incluye tambien en entorno de anaconda con python 3.8 para ejecutar todos los scripts

Preprocesamiento de la data:
- Dado un directorio con el dataset FER2013 o RAFDB con directorios hijos de train, test y validation se pueden ejecutar los scripts de preprocesameinto
- rename.py renombra las imagenes para tener un estandar en las carpetas
- data_csv.py crea un csv con los nomrbes de los registros renombrados y genera un csv que el modelo puede leer para identificar las imagenes y la clase a las que pertenecen

Entrenamiento del modelo:
- Una vez formateada la data se le puede pasar al modelo la ruta del directorio que incluye los directorios train, test y validation para el entrenamiento
- Cuando el entrenameinto finaliza se descargar el .pth de los mejores resultados obtenidos y se crea un archivo csv con las metricas por epoca

Resultados:
- Se adjuntan las matrices de confusion de los mejores resultados obtenidos
- El script de visualizar resultados utiliza el .pth para obtener los mejores parametros y la data previamente usada para imprimir las metricas y la matriz de confusión
