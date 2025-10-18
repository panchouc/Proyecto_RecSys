import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def resumen_juegos(df: pl.DataFrame):
    return df["Game Title"].n_unique()

def resumen_categorias(df: pl.DataFrame):
    return {
        "n_genres": df["Genre"].n_unique(),
        "n_platforms": df["Platform"].n_unique(),
        "n_developers": df["Developer"].n_unique()
    }

def revisar_duplicados(df: pl.DataFrame, subset: list[str] = None):
    total = df.height
    if subset is None:
        n_unicos = df.unique().height
    else:
        n_unicos = df.unique(subset=subset).height
    duplicados = total - n_unicos
    return {
        "total_filas": total,
        "duplicados": duplicados,
        "pct": duplicados / total if total > 0 else 0
    }


def parametrizar_rating_a_5(df: pl.DataFrame, col: str = "User Rating"):
    vals = df[col].to_numpy().astype(float)

    minimo = np.min(vals)
    maximo = np.max(vals)

    scaled = 1 + ( (vals - minimo) / (maximo - minimo) ) * (5 - 1)

    return df.with_columns(pl.Series(f"{col}_1a5", scaled))

def matriz_correlacion(data_numerica, target_col='User Rating', savefig_name=None, top_n=20):
    """Matriz de correlación con las variables más correlacionadas con la variable objetivo"""
    
    # Validar que existe la columna objetivo
    if target_col not in data_numerica.columns:
        available_cols = [col for col in data_numerica.columns if 'rating' in col.lower()]
        if available_cols:
            target_col = available_cols[0]
            print(f"Usando '{target_col}' como columna objetivo")
        else:
            print("Columnas disponibles:", data_numerica.columns[:10])
            raise ValueError(f"No se encontró '{target_col}' ni columnas similares")
    
    df_corr = data_numerica.to_pandas()
    df_corr = df_corr.loc[:, df_corr.var() != 0]
    correlaciones = df_corr.corr()[target_col].abs().sort_values(ascending=False)
    
    correlaciones = correlaciones.dropna()
    top_vars = correlaciones.head(top_n).index.tolist()
    
    if len(top_vars) < 2:
        print(f"Solo {len(top_vars)} variables válidas encontradas")
        return correlaciones
    
    corr_matrix = df_corr[top_vars].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  
    
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True, 
                fmt='.2f', 
                cbar_kws={'shrink': 0.8},
                linewidths=0.5)
    
    plt.title(f'Matriz de Correlación - Top {len(top_vars)} Variables\nmás correlacionadas con {target_col}')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if savefig_name:
        plt.savefig(savefig_name, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nTop 10 correlaciones con {target_col}:")
    for var, corr in correlaciones.head(10).items():
        if var != target_col:
            print(f"  {var}: {corr:.3f}")
    
    return correlaciones