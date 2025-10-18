import polars as pl

# https://github.com/pola-rs/polars/issues/5643

NUMERIC_POLARS_DTYPES = [
    pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64, 
]