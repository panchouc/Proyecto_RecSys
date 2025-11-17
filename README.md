# **Proyecto**
### **Multimodalidad: Recomendación deVideojuegos**
### Sistemas Recomendadores IIC3633-1 2025-2
### **Grupo 3:** 

- Nicolás Antonio Bueno Abett de la Torre 

- Felipe Andrés Fuentes González

- Jorge Andrés Jacque Palma

- Francisco Nicolás Solís Gormaz

### Organización de archivos

El repositorio tiene 7 carpetas principales:

- `data`: Contiene `sampled_recommendations.csv`, que es el muestreo realizado del set de interacciones original `recommendations.csv` debido a su gran tamaño (decenas de millones de interacciones y un peso de cerca de 2 GB). El código con el que se realizó el muestreo está en `muestreo.ipynb`, en donde se detalla el procedimiento. Además, esta carpeta también contiene la carpeta `split`, en donde se encuentran los archivos .csv de los sets de entrenamiento, testeo, y validación obtenidos a partir de la división del muestreo del dataset original, es decir, el split se hizo sobre `sampled_recommendations.csv`. Todos los modelos usarán estos mismos archivos de entrenamiento, testeo, y validación.

- `data_analysis`: Contiene los scripts y notebooks utilizados para el análisis de datos, incluyendo limpieza, transformación y visualización de datos. El archivo principal es `nuevo_analisis.ipynb`, que contiene el análisis del nuevo dataset utilizado después del hito 2. Los otros archivos corresponden al análisis del dataset antiguo: el archivo a ejecutar era `main.py`, `plotting.py` contiene las funciones de visualización y generación de gráficos, `funciones.py` contiene funciones para cargar y tratar los datos, `parametros.py` tiene parámetros utilizados en el resto de archivos, y la carpeta `figures/` contiene imágenes de los gráficos generados.

- `modelos_ref`: Contiene los 4 modelos de referencia utilizados (Random, Most Popular, User KNN, Item KNN). El archivo principal es `modelos_referenciales.ipynb`.

- `modelo_lightgcn`: Contiene el modelo desarrollado a base de LightGCN. El archivo principal es `LightGCN_V2.ipynb`, mientras que `LightGCN.ipynb` corresponde al código antiguo del hito 2.

- `modelo_lightfm`: Contiene el modelo LightFM desarrollado. Los archivos principales son `LightFM_base.ipynb`, `LightFM_features_cat.ipynb`, y `LightFM_features_num.ipynb`.

- `modelo_deepfm`: Contiene el modelo desarrollado a base de DeepFM. El archivo principal es `DeepFM_V2.ipynb`, mientras que `DeepFM.ipynb` corresponde al código antiguo del hito 2.

- `modelo_multivae`: Contiene el modelo desarrollado a base de MultiVAE. El archivo principal es `MultiVAE.ipynb`.

### Dataset principal: 

"Game Recommendations on Steam: A dataset of games, users and reviews for building recommendation systems". Anton Kozyriev, 2024. Kaggle. Link: https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam?select=recommendations.csv

Contiene:

- `recommendations.csv`: evaluaciones de usuarios a ítems.

- `users.csv`: información general de usuarios registrados.

- `games.csv`: información general de videojuegos y add-ons disponibles en la plataforma.

- `games_metadata.json`: información adicional de videojuegos y add-ons disponibles en la plataforma (metadata).

### Datasets adicionales:

- Metadata adicional de videojuegos de Steam: "Steam Store Games (Clean dataset): Combined data of 27,000 games scraped from Steam and SteamSpy APIs". Nik Davis, 2019. Kaggle. Link: https://www.kaggle.com/datasets/nikdavis/steam-store-games?select=steam.csv

