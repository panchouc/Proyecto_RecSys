import polars as pl
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA



from plotting import graficar_histograma
from parametros import NUMERIC_POLARS_DTYPES

data = pl.read_csv("video_game_reviews.csv")
nombres_unicos = data["Game Title"].unique().to_list()

nombre_to_id = {nombre: numero for numero, nombre in enumerate(nombres_unicos)}

data = (data.with_columns(pl.col("Game Title").replace_strict(nombre_to_id).alias("id")))

columnas_histogram = ["User Rating", "Price"]

columnas_dummies = ["Developer", "Genre", "Age Group Targeted", "Platform", "User Review Text"]


# Poner referencia
data = data.to_dummies(columnas_dummies, separator="_")
new_cols = {}
for col in data.columns:
    for base in columnas_dummies:
        if col.startswith(base + "_"):
            new_cols[col] = col.replace(base + "_", "")

data = data.rename(new_cols)

data_numerica = data.select(pl.col(NUMERIC_POLARS_DTYPES)).drop("id")

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_numerica)

inertias = []
valor_k = []
silhouettes = list()
calcular_k = False


# MEJOR PASEMOS LA DATA CON PCA A 2D Y AHI HACEMOS EL KMEANS PA PODER GRAFICARLO



if calcular_k:
    for k in range(2, 9):
        print(f"Valor k = {k}")
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(scaled_data)
        inertias.append(kmeans.inertia_)
        valor_k.append(k)
        silhouettes.append(silhouette_score(scaled_data, kmeans.labels_))

    plt.figure()
    plt.plot(valor_k, silhouettes, marker="o")
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.title("Silhouette por k")
    plt.show()

kmeans = KMeans(n_clusters=9)
kmeans.fit(scaled_data)



#graficar_histograma(data, "User Rating")
