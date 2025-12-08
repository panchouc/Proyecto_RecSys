# %% [markdown]
# # XGBoost

# %%
import os
import json
import pandas as pd
import numpy as np

from xgboost import XGBRanker
from collections import defaultdict
from sentence_transformers import SentenceTransformer

def precision_at_k(rec_k, rel_set):
    if len(rec_k) == 0:
        return 0.0
    hits = sum((i in rel_set) for i in rec_k)
    return hits / len(rec_k)

def recall_at_k(rec_k, rel_set):
    if len(rel_set) == 0:
        return 0.0
    hits = sum((i in rel_set) for i in rec_k)
    return hits / len(rel_set)

def ndcg_at_k(rec_k, rel_set):
    if len(rec_k) == 0:
        return 0.0
    dcg = 0.0
    for rank, it in enumerate(rec_k, start=1):
        if it in rel_set:
            dcg += 1.0 / np.log2(rank + 1)
    ideal = min(len(rel_set), len(rec_k))
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal + 1))
    return (dcg / idcg) if idcg > 0 else 0.0

def hit_score_at_k(rec_k, rel_set):
    cant_relevantes = set(rec_k).intersection(set(rel_set))
    return len(cant_relevantes)

def map_at_k(rec_k, rel_set):
    if len(rec_k) == 0:
        return 0.0
    ap_sum = 0.0
    hits = 0
    for rank, it in enumerate(rec_k, start=1):
        if it in rel_set:
            hits += 1
            ap_sum += hits / rank
    return ap_sum / len(rel_set) if len(rel_set) > 0 else 0.0

def diversity_at_k(rec_k, info_videojuegos):
    generos_total = set()
    for app_id in rec_k:
        for genero in info_videojuegos[app_id]:
            generos_total.add(genero)
    if not generos_total:
        return 0
    return len(generos_total)

def f1_at_k(rec_k, rel_set):
    if len(rec_k) == 0 or len(rel_set) == 0:
        return 0.0
    p = precision_at_k(rec_k, rel_set)
    r = recall_at_k(rec_k, rel_set)
    if (p + r) <= 0:
        return 0.0
    return 2 * p * r / (p + r)

base_dir = os.getcwd()
data_dir = os.path.join(base_dir, "..", "data", "split")

ruta_train = os.path.join(data_dir, "train_split.csv")
ruta_test = os.path.join(data_dir, "test_split.csv")
ruta_val = os.path.join(data_dir, "val_split.csv")  # por si lo usas después

ruta_metadata = os.path.join("games_metadata.json")

train_set = pd.read_csv(ruta_train)
test_set  = pd.read_csv(ruta_test)

train_set["app_id"] = train_set["app_id"].astype(int)
test_set["app_id"]  = test_set["app_id"].astype(int)

train_set["hours"] = np.log1p(train_set["hours"])
test_set["hours"]  = np.log1p(test_set["hours"])

regla_rating = {True: 1, False: 0}
train_set["rating"] = train_set["is_recommended"].map(regla_rating)
test_set["rating"]  = test_set["is_recommended"].map(regla_rating)

items_relevantes = (
    test_set[test_set["rating"] == 1]
    .groupby("user_id")["app_id"]
    .apply(list)
    .to_dict()
)

final_dict = {}
info_videojuegos = defaultdict(list)
set_tags = set()

with open(ruta_metadata, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        app_id = int(obj["app_id"])  # forzamos int

        final_dict[app_id] = str(obj["description"])
        info_videojuegos[app_id].extend(obj["tags"])

        for tag in obj["tags"]:
            set_tags.add(tag)

with open("info_videojuegos.json", "w", encoding="utf-8") as f:
    json.dump(info_videojuegos, f, indent=4, ensure_ascii=False)

descripciones = list(final_dict.values())
keys_app_id   = list(final_dict.keys())  # alineado con descripciones
keys_app_id   = [int(a) for a in keys_app_id]

print("Generando embeddings MiniLM...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(descripciones)  # shape: (n_items, dim)

app_id_to_idx = {app_id: idx for idx, app_id in enumerate(keys_app_id)}

valid_app_ids = set(app_id_to_idx.keys())
train_set = train_set[train_set["app_id"].isin(valid_app_ids)].copy()
test_set  = test_set[test_set["app_id"].isin(valid_app_ids)].copy()

train_set = train_set.sort_values("user_id").reset_index(drop=True)
test_set  = test_set.sort_values("user_id").reset_index(drop=True)

items_relevantes = (
    test_set[test_set["rating"] == 1]
    .groupby("user_id")["app_id"]
    .apply(list)
    .to_dict()
)


train_app_idx = train_set["app_id"].map(app_id_to_idx).values
emb_train     = embeddings[train_app_idx]  # (n_train, dim)

user_train = train_set["user_id"].values.reshape(-1, 1)
app_train  = train_set["app_id"].values.reshape(-1, 1)

X_train = np.hstack([user_train, app_train, emb_train])
y_train = train_set[["hours"]].values  # objetivo: horas (log1p)

# Grupos de ranking por usuario
group_train = train_set.groupby("user_id").size().tolist()

ranker = XGBRanker(
    objective="rank:pairwise",
    learning_rate=0.1,
    n_estimators=300,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

print("Entrenando XGBRanker...")
ranker.fit(
    X_train,
    y_train,
    group=group_train
)

print("Generando recomendaciones por usuario...")

recomendaciones_def = defaultdict(list)

# Usuarios y candidatos
test_users      = test_set["user_id"].unique()
candidate_items = np.sort(test_set["app_id"].unique())  # puedes cambiar a todos los items si quieres

# Precomputar índices de candidatos en la matriz de embeddings
candidate_idx = np.array([app_id_to_idx[a] for a in candidate_items])
emb_cand_all  = embeddings[candidate_idx]  # (n_candidates, dim)

for user_id in test_users:
    # Columnas numéricas
    n_cand = len(candidate_items)
    user_col = np.full(n_cand, user_id).reshape(-1, 1)
    app_col  = candidate_items.reshape(-1, 1)

    # Mismos embeddings para todos los usuarios (ya precomputados)
    X_test_user = np.hstack([user_col, app_col, emb_cand_all])

    # Predicción para este usuario
    y_pred_user = ranker.predict(X_test_user)

    # Ordenar candidatos por score descendente
    order = np.argsort(-y_pred_user)
    recs_ordenadas = candidate_items[order].tolist()
    recomendaciones_def[user_id] = recs_ordenadas


precision_list = []
recall_list = []
ndcg_list = []
f1_list = []
hitrate_list = []
map10_list = []
diversity_list = []

for usuario, recomendaciones_usuario in recomendaciones_def.items():
    # Si el usuario no tiene relevantes, lo saltamos o lo contamos con métricas 0
    if usuario not in items_relevantes:
        continue

    items_rel_usuario = items_relevantes[usuario]
    recomendaciones_10 = recomendaciones_usuario[:10]

    precision_usuario = precision_at_k(recomendaciones_10, items_rel_usuario)
    recall_usuario    = recall_at_k(recomendaciones_10, items_rel_usuario)
    ndcg_usuario      = ndcg_at_k(recomendaciones_10, items_rel_usuario)
    f1_usuario        = f1_at_k(recomendaciones_10, items_rel_usuario)
    hitrate_usuario   = hit_score_at_k(recomendaciones_10, items_rel_usuario)
    map10_usuario     = map_at_k(recomendaciones_10, items_rel_usuario)
    diversity_usuario = diversity_at_k(recomendaciones_10, info_videojuegos)

    precision_list.append(precision_usuario)
    recall_list.append(recall_usuario)
    ndcg_list.append(ndcg_usuario)
    f1_list.append(f1_usuario)
    hitrate_list.append(hitrate_usuario)
    map10_list.append(map10_usuario)
    diversity_list.append(diversity_usuario)

print(f"Precision@10: {np.mean(precision_list):.4f}")
print(f"Recall@10:    {np.mean(recall_list):.4f}")
print(f"nDCG@10:      {np.mean(ndcg_list):.4f}")
print(f"F1-Score@10:  {np.mean(f1_list):.4f}")
print(f"Hit Score@10: {np.mean(hitrate_list):.4f}")
print(f"MAP@10:       {np.mean(map10_list):.4f}")
print(f"Diversity:    {np.mean(diversity_list):.4f}")
