# **Proyecto**
### **Multimodalidad: Recomendación de Videojuegos**
### Sistemas Recomendadores IIC3633-1 2025-2
### **Grupo 3:** 

- Nicolás Antonio Bueno Abett de la Torre 

- Felipe Andrés Fuentes González

- Jorge Andrés Jacque Palma

- Francisco Nicolás Solís Gormaz

### Organización de archivos

El repositorio tiene 6 carpetas principales:

- `data_analysis`: Contiene los scripts y notebooks utilizados para el análisis de datos, incluyendo limpieza, transformación y visualización de datos. El archivo principal es `main.py`. Se debe ejecutar este archivo para reproducir el análisis de datos. `plotting.py` contiene las funciones de visualización y generación de gráficos utilizadas en el análisis. `funciones.py` contiene funciones para cargar y tratar los datos. `parametros.py` tiene parámetros utilizadas en el resto de archivos. La carpeta `figures` contiene imágenes de los gráficos generados en el análisis de datos.

- `modelos_ref`: Contiene los 4 modelos de referencia utilizados (Random, Most Popular, User KNN, Item KNN). El archivo principal es `modelos_referenciales.ipynb`.

- `modelo_lightgcn`: Contiene el modelo desarrollado a base de LightGCN. El archivo principal es `modelo_lightgcn.ipynb`.

- `modelo_lightfm`: Contiene el modelo LightFM desarrollado. Los archivos principales son `LightFM_base.ipynb`, `LightFM_features_cat.ipynb`, y `LightFM_features_num.ipynb`.

- `modelo_deepfm`: Contiene el modelo desarrollado a base de DeepFM. El archivo principal es `DeepFM.ipynb`.

- `modelo_multivae`: Contiene el modelo desarrollado a base de MultiVAE. El archivo principal es `XXX.ipynb`.

### Dataset principal: 

"Video Game Reviews and Rating: A Randomly Generated Dataset to Help Practice Machine Learning Skills". Jahnavi Paliwal, 2024. Kaggle.

### Links para descargar datasets:

- Dataset principal: https://www.kaggle.com/datasets/jahnavipaliwal/video-game-reviews-and-ratings

- Datasets de información adicional: https://www.kaggle.com/discussions/general/332936
