import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import polars as pl


def graficar_histograma(data, column, savefig_name=None):
    sns.histplot(data=data, x=column)

    if savefig_name:
        plt.savefig(savefig_name)

    plt.show()

def graficar_clusters_pca(X_pca, labels, savefig_name=None):
    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="tab10", s=10, alpha=0.7)
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.title("Clusters de usuarios en espacio PCA")
    plt.colorbar(label="Cluster")

    if savefig_name:
        plt.savefig(savefig_name, dpi=300, bbox_inches='tight')

    plt.show()
    

def graficar_cluster_optimo(k_range, inertias, silhouette_scores, best_k, savefig_name=None):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inercia')
    plt.title('Método del Codo')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, 'ro-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score por k')
    plt.grid(True, alpha=0.3)

    # Marcar el mejor k según silhouette
    plt.axvline(x=best_k, color='g', linestyle='--', alpha=0.7, 
                label=f'Mejor k={best_k}')
    plt.legend()

    plt.tight_layout()
    
    if savefig_name:
        plt.savefig(savefig_name, dpi=300, bbox_inches='tight')
    
    plt.show()

# ------ Graficos analisis exploratorio ------- #
def hist_num(df: pl.DataFrame, col: str, log=False, winsor_p=0.0, bins=40, savefig_name=None, show=True):
    s = df[col].drop_nulls().to_numpy()
    if winsor_p > 0:
        lo, hi = np.quantile(s, [winsor_p, 1-winsor_p])
        s = np.clip(s, lo, hi)
    if log:
        s = s[s > 0]
        s = np.log10(s)
        plt.xlabel(f"log10({col})")
    else:
        plt.xlabel(col)
    plt.hist(s, bins=bins)
    plt.title(f"Histograma: {col}")
    plt.ylabel("Frecuencia")
    if savefig_name:
        plt.savefig(savefig_name)
    if show:
        plt.show()

def topk_cat_count(df: pl.DataFrame, col: str, k: int = 10, as_pct: bool = False, savefig_name=None, show=True):

    if col not in df.columns:
        raise ValueError(f"La columna '{col}' no existe en el DataFrame.")

    tabla = (df.group_by(col)
               .len()
               .rename({"len": "count"})
               .sort("count", descending=True)
               .head(k))

    total = float(df.height) if df.height else 1.0
    tabla = tabla.with_columns((pl.col("count") / total).alias("pct"))

    labels = [str(x) for x in tabla[col].to_list()]
    values = tabla["pct"].to_numpy() if as_pct else tabla["count"].to_numpy()

    # --- figura nueva para evitar “mezcla” con gráficos previos ---
    plt.figure(figsize=(10, max(5, 0.5*len(labels))))
    plt.barh(labels, values)
    plt.gca().invert_yaxis()  # que el top quede arriba
    if as_pct:
        plt.xlabel("Porcentaje")
    else:
        plt.xlabel("Conteo")
    plt.title(f"Top {len(labels)} de {col}")
    plt.tight_layout()

    if savefig_name:
        plt.savefig(savefig_name)
    if show:
        plt.show()
    return tabla


# Ejemplo: Price vs Rating
def scatter_price_vs_rating(df: pl.DataFrame,
                            price_col: str = "Price",
                            rating_col: str = "User Rating",
                            alpha: float = 0.3,
                            s: int = 20,
                            savefig_name=None, show=True):
    """
    Scatter plot de Precio vs Rating de usuarios.
    """
    if price_col not in df.columns or rating_col not in df.columns:
        raise ValueError("Revisa los nombres de columnas (Price, User Rating).")

    x = df[price_col].to_numpy()
    y = df[rating_col].to_numpy()

    plt.figure(figsize=(10,6))
    plt.scatter(x, y, alpha=alpha, s=s, c="tab:blue")
    plt.xlabel("Precio")
    plt.ylabel("User Rating")
    plt.title("Precio vs User Rating")
    plt.grid(alpha=0.3)
    if savefig_name:
        plt.savefig(savefig_name)
    if show:
        plt.show()

def avg_price_by_platform(df: pl.DataFrame, top_k: int = 10, savefig_name=None, show=True):
    # Agrupar por plataforma y calcular precio promedio
    tabla = (
        df.group_by("Platform")
          .agg(pl.col("Price").mean().alias("avg_price"),
               pl.len().alias("n"))
          .sort("avg_price", descending=True)
          .head(top_k)
    )

    platforms = tabla["Platform"].to_list()
    avg_price = tabla["avg_price"].to_numpy()

    # Gráfico de barras
    plt.figure(figsize=(8, 5))
    plt.bar(platforms, avg_price, color="steelblue")
    plt.ylabel("Precio promedio")
    plt.xlabel("Plataforma")
    plt.title("Precio promedio por Plataforma")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if savefig_name:
        plt.savefig(savefig_name)
    if show:
        plt.show()

    return tabla

def avg_num_by_cat(df, num_col: str, cat_col: str, top_k: int = 10, savefig_name=None, show=True):
    tabla = (df.group_by(cat_col)
               .agg(pl.col(num_col).mean().alias("avg"),
                    pl.len().alias("n"))
               .sort("avg", descending=True)
               .head(top_k))

    avg = tabla["avg"].to_numpy()
    cats = [str(x) for x in tabla[cat_col].to_list()]

    plt.figure(figsize=(10, 6))
    plt.barh(cats, avg, color="steelblue")
    plt.xlabel(f"Promedio {num_col}")
    plt.ylabel(cat_col)
    plt.title(f"Promedio {num_col} por {cat_col} (Top {len(cats)})")
    plt.gca().invert_yaxis()  # para que el mayor quede arriba
    plt.tight_layout()
    if savefig_name:
        plt.savefig(savefig_name)
    if show:
        plt.show()
    return tabla