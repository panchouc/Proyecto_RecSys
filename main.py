import polars as pl
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import numpy as np

from plotting import (graficar_histograma, graficar_clusters_pca, graficar_cluster_optimo, hist_num, topk_cat_count, scatter_price_vs_rating,
                       avg_num_by_cat, avg_price_by_platform)

from parametros import NUMERIC_POLARS_DTYPES
import funciones

data = pl.read_csv("video_game_reviews.csv")

# ----- Analisis General de Datos en General -------- # 
print("Juegos distintos:", funciones.resumen_juegos(data))
print("Categorías:", funciones.resumen_categorias(data))
print("\nDuplicados (todas las columnas):")
print(funciones.revisar_duplicados(data))

hist_num(data, "User Rating", savefig_name='figures/hist_user_rating.png', show=False)
topk_cat_count(data, "Genre", k=10, savefig_name='figures/topk_genre.png', show=False)
scatter_price_vs_rating(data, "Price", "User Rating", savefig_name='figures/scatter_price_user_rating.png', show=False)
avg_price_by_platform(data, savefig_name='figures/avg_price_platform.png', show=False)
avg_num_by_cat(data, "User Rating", "Genre", savefig_name='figures/avg_user_rating_genre.png', show=False)

# Tabla resumen por género
tabla = (
    data.group_by("Genre")
      .agg([
          pl.len().alias("n"),
          pl.col("User Rating").mean().alias("mean"),
          pl.col("User Rating").median().alias("median"),
          pl.col("User Rating").std(ddof=0).alias("std"),
          pl.col("User Rating").quantile(0.25).alias("q25"),
          pl.col("User Rating").quantile(0.75).alias("q75"),
      ])
      .sort("mean", descending=True)
)
print(tabla)

# Interacciones: promedio por (Género, Plataforma)
gp = (
    data.group_by(["Genre","Platform"])
      .agg(pl.col("User Rating").mean().alias("avg"))
      .pivot(values="avg", index="Genre", columns="Platform")
      .fill_null(0)
)
print(gp)

# ----- Preparación para Clustering -------- #

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
calcular_pca = False

# ------- Intento de PCA ---------- # 
# MEJOR PASEMOS LA DATA CON PCA A 2D Y AHI HACEMOS EL KMEANS PA PODER GRAFICARLO
if calcular_pca:
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(scaled_data)
    silhouette_scores = []
    inertias = []
    print("Probando diferentes valores de k...")
    max_k = 6
    k_range = range(2, max_k + 1)
    for k in k_range:
        print(f"  k = {k}...", end=" ")
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels_full = kmeans.fit_predict(X_pca)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_pca, labels_full))
        print("✓")
        
    best_k = k_range[np.argmax(silhouette_scores)]
    graficar_cluster_optimo(k_range, inertias, silhouette_scores, best_k, savefig_name='figures/cluster_optimization_pca.png')

    print(f"Mejor número de clusters según Silhouette Score: {best_k}")
    print(f"Silhouette Score máximo: {max(silhouette_scores):.4f}")

    kmeans = KMeans(n_clusters=best_k, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(X_pca)
    graficar_clusters_pca(X_pca, labels, savefig_name='figures/PCA_bestk.png')


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

