import polars as pl
import matplotlib.pyplot as plt
import os

from plotting import graficar_histograma

data = pl.read_csv("video_game_reviews.csv")
nombres_unicos = data["Game Title"].unique().to_list()

nombre_to_id = {nombre: numero for numero, nombre in enumerate(nombres_unicos)}

data = (data.with_columns(pl.col("Game Title").replace_strict(nombre_to_id).alias("id")))

#print(data)

columnas_histogram = ["User Rating", "Price"]



#print(nombre_to_id)
print(data.columns)


graficar_histograma(data, "User Rating")
