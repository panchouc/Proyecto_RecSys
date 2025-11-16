import argparse
import os
import re
import sys
import unicodedata
from typing import Tuple

import pandas as pd


def _strip_accents(text: str) -> str:
	if not isinstance(text, str):
		return ""
	text = unicodedata.normalize("NFKD", text)
	return "".join(ch for ch in text if not unicodedata.combining(ch))


_TRADEMARKS_RE = re.compile(r"[\u2122\u00AE\u00A9]")  # ™ ® ©
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_title(s: str) -> str:
	"""
	Normaliza un título de juego para emparejar entre datasets:
	- Minúsculas, sin acentos
	- Elimina símbolos de marca (™, ®, ©) y puntuación
	- Colapsa espacios múltiples
	"""
	if s is None:
		return ""
	s = str(s).strip()
	s = _strip_accents(s)
	s = s.lower()
	s = _TRADEMARKS_RE.sub("", s)
	s = _NON_ALNUM_RE.sub(" ", s)
	s = re.sub(r"\s+", " ", s).strip()
	return s


def load_games_csv(path: str) -> pd.DataFrame:
	df = pd.read_csv(path)
	if "title" not in df.columns:
		raise ValueError(f"El CSV de games no contiene la columna 'title': {path}")
	df = df.copy()
	df["title_key"] = df["title"].map(normalize_title)
	return df


def load_steam_csv(path: str) -> pd.DataFrame:
	df = pd.read_csv(path)
	# steam.csv de Kaggle suele traer 'name' como nombre del juego
	if "name" not in df.columns:
		raise ValueError(f"El CSV de steam no contiene la columna 'name': {path}")
	df = df.copy()
	df["name_key"] = df["name"].map(normalize_title)
	# Evitar ambigüedades muchas-a-muchas en la unión conservando el primer match por key
	df = df.sort_index().drop_duplicates(subset=["name_key"], keep="first")
	return df


def merge_games_steam(
	games_df: pd.DataFrame,
	steam_df: pd.DataFrame,
	how: str = "left",
) -> Tuple[pd.DataFrame, int, int]:
	"""
	Une games (title) con steam (name) usando claves normalizadas.
	Devuelve (df_merged, n_matched, n_left).
	"""
	left_count = len(games_df)
	merged = games_df.merge(
		steam_df,
		left_on="title_key",
		right_on="name_key",
		how=how,
		suffixes=("_games", "_steam"),
	)
	matched = int(merged["name"].notna().sum()) if "name" in merged.columns else 0
	return merged, matched, left_count


def _default_paths(base_dir: str) -> Tuple[str, str, str]:
	games_path = os.path.join(base_dir, "Game Recommendations on Steam", "games.csv")
	steam_path = os.path.join(base_dir, "Steam Store Games", "steam.csv")
	# Nombre por defecto deja explícito que solo incluye coincidencias
	out_path = os.path.join(base_dir, "games_steam_matched.csv")
	return games_path, steam_path, out_path


def main(argv=None) -> int:
	base_dir = os.path.dirname(os.path.abspath(__file__))
	def_games, def_steam, def_out = _default_paths(base_dir)

	parser = argparse.ArgumentParser(
		description="Une games.csv (title) con steam.csv (name) por nombre normalizado",
	)
	parser.add_argument("--games", default=def_games, help="Ruta a games.csv")
	parser.add_argument("--steam", default=def_steam, help="Ruta a steam.csv")
	parser.add_argument("--out", default=def_out, help="Ruta del CSV de salida")
	parser.add_argument(
		"--how",
		default="inner",
		choices=["left", "inner", "right", "outer"],
		help="Tipo de merge (por defecto inner: solo coincidencias entre juegos)",
	)
	args = parser.parse_args(argv)

	if not os.path.exists(args.games):
		print(f"No se encontró games.csv en: {args.games}", file=sys.stderr)
		return 2
	if not os.path.exists(args.steam):
		print(f"No se encontró steam.csv en: {args.steam}", file=sys.stderr)
		return 2

	games_df = load_games_csv(args.games)
	steam_df = load_steam_csv(args.steam)
	merged, matched, total = merge_games_steam(games_df, steam_df, how=args.how)

	os.makedirs(os.path.dirname(args.out), exist_ok=True)
	merged.to_csv(args.out, index=False)

	coverage = (matched / total * 100.0) if total else 0.0
	print(
		f"Filas games: {total} | Matcheadas: {matched} ({coverage:.2f}%) | Salida: {args.out}"
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

