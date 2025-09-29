import polars as pl

def resumen_juegos(df: pl.DataFrame):
    """Cantidad de juegos únicos"""
    return df["Game Title"].n_unique()

def resumen_categorias(df: pl.DataFrame):
    """Cuántos géneros, plataformas y developers distintos"""
    return {
        "n_genres": df["Genre"].n_unique(),
        "n_platforms": df["Platform"].n_unique(),
        "n_developers": df["Developer"].n_unique()
    }

def revisar_duplicados(df: pl.DataFrame, subset: list[str] = None):
    """¿Hay reseñas duplicadas?"""
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
