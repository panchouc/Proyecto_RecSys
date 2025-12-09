
import os
import json
import random
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRanker



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
    # return 1.0 if any((i in rel_set) for i in rec_k) else 0.0


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
        return 0.0
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
ruta_val = os.path.join(data_dir, "val_split.csv")

ruta_metadata = os.path.join("games_metadata.json")

train_set = pd.read_csv(ruta_train)
test_set = pd.read_csv(ruta_test)
val_set = pd.read_csv(ruta_val)

train_set["hours"] = np.log1p(train_set["hours"])
test_set["hours"] = np.log1p(test_set["hours"])
val_set["hours"] = np.log1p(val_set["hours"])

regla_rating = {True: 1, False: 0}
train_set['rating'] = train_set['is_recommended'].map(regla_rating)
test_set['rating'] = test_set['is_recommended'].map(regla_rating)
val_set['rating'] = val_set['is_recommended'].map(regla_rating)

items_relevantes_test = test_set.groupby("user_id")["app_id"].apply(list).to_dict()
items_relevantes_val = val_set.groupby("user_id")["app_id"].apply(list).to_dict()



final_dict = {}
info_videojuegos = defaultdict(list)

set_tags = set()
with open(ruta_metadata, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        app_id = obj["app_id"]
        final_dict[app_id] = str(obj["description"])
        info_videojuegos[app_id].extend(obj["tags"])

        for tag in obj["tags"]:
            set_tags.add(tag)

with open("info_videojuegos.json", "w", encoding="utf-8") as f:
    json.dump(info_videojuegos, f, indent=4, ensure_ascii=False)

descripciones = list(final_dict.values())
keys_app_id = list(final_dict.keys())

train_set = train_set.sort_values("user_id").reset_index(drop=True)
test_set = test_set.sort_values("user_id").reset_index(drop=True)
val_set = val_set.sort_values("user_id").reset_index(drop=True)

vectorizer = TfidfVectorizer(stop_words="english")
descripciones_train_tfid = vectorizer.fit_transform(descripciones)
dict_transformados = {i: j for i, j in zip(keys_app_id, descripciones_train_tfid)}

train_set["descripciones"] = train_set["app_id"].map(dict_transformados)
test_set["descripciones"] = test_set["app_id"].map(dict_transformados)
val_set["descripciones"] = val_set["app_id"].map(dict_transformados)

columnas_importantes = ["user_id", "app_id", "hours", "descripciones"]
train_set = train_set[columnas_importantes]
test_set = test_set[columnas_importantes]
val_set = val_set[columnas_importantes]


test_set_user_uniques = test_set["user_id"].unique().tolist()
test_set_app_uniques = test_set["app_id"].unique().tolist()

cartesian_df = pd.DataFrame(
    list(product(test_set_user_uniques, test_set_app_uniques)),
    columns=["user_id", "app_id"]
)
cartesian_df["descripciones"] = cartesian_df["app_id"].map(dict_transformados)

val_user_uniques = val_set["user_id"].unique().tolist()
val_app_uniques = val_set["app_id"].unique().tolist()

cartesian_val = pd.DataFrame(
    list(product(val_user_uniques, val_app_uniques)),
    columns=["user_id", "app_id"]
)
cartesian_val["descripciones"] = cartesian_val["app_id"].map(dict_transformados)



columnas_train, columnas_predict = ["user_id", "app_id", "descripciones"], ["hours"]

X_train_df = train_set[columnas_train]
y_train = train_set[columnas_predict].values.ravel()

X_val_df = cartesian_val[columnas_train]
X_test_df = cartesian_df[columnas_train]

numerical_features_train = X_train_df[["user_id", "app_id"]].values
numerical_features_val = X_val_df[["user_id", "app_id"]].values
numerical_features_test = X_test_df[["user_id", "app_id"]].values

descripciones_sparse_train = scipy.sparse.vstack(X_train_df["descripciones"].tolist())
descripciones_sparse_val = scipy.sparse.vstack(X_val_df["descripciones"].tolist())
descripciones_sparse_test = scipy.sparse.vstack(X_test_df["descripciones"].tolist())

X_train = scipy.sparse.hstack((numerical_features_train, descripciones_sparse_train))
X_val = scipy.sparse.hstack((numerical_features_val, descripciones_sparse_val))
X_test = scipy.sparse.hstack((numerical_features_test, descripciones_sparse_test))

group_train = train_set.groupby("user_id").size().tolist()


def evaluar_ndcg_en_validacion(params):

    ranker_tmp = XGBRanker(
        objective="rank:pairwise",
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )

    ranker_tmp.fit(
        X_train,
        y_train,
        group=group_train
    )

    y_val_pred = ranker_tmp.predict(X_val)

    recomendaciones_val = defaultdict(list)
    for i in range(X_val_df.shape[0]):
        user_id = X_val_df.iloc[i]['user_id']
        app_id = X_val_df.iloc[i]['app_id']
        score = y_val_pred[i]
        recomendaciones_val[user_id].append((app_id, score))

    recomendaciones_val_def = {}
    for usuario, lista_rec in recomendaciones_val.items():
        lista_ord = sorted(lista_rec, key=lambda x: x[1], reverse=True)
        recomendaciones_val_def[usuario] = [app for app, _ in lista_ord]

    ndcg_list_val = []
    for usuario, recs_usuario in recomendaciones_val_def.items():
        rel_items = items_relevantes_val.get(usuario, [])
        if len(rel_items) == 0:
            continue
        recs_10 = recs_usuario[:10]
        ndcg_usuario = ndcg_at_k(recs_10, rel_items)
        ndcg_list_val.append(ndcg_usuario)

    if len(ndcg_list_val) == 0:
        return 0.0

    return float(np.mean(ndcg_list_val))



param_grid = {
    "learning_rate": [0.03, 0.05, 0.1, 0.2],
    "n_estimators": [200, 300, 500],
    "max_depth": [4, 6, 8],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}


print("Combis")
todas_combis = []
for lr in param_grid["learning_rate"]:
    for n_est in param_grid["n_estimators"]:
        for md in param_grid["max_depth"]:
            for ss in param_grid["subsample"]:
                for cs in param_grid["colsample_bytree"]:
                    todas_combis.append({
                        "learning_rate": lr,
                        "n_estimators": n_est,
                        "max_depth": md,
                        "subsample": ss,
                        "colsample_bytree": cs,
                    })

random.shuffle(todas_combis)
num_candidatos = 20  
candidatos = todas_combis[:num_candidatos]

mejor_ndcg = -1.0
mejores_params = None

for i, params in enumerate(candidatos, start=1):
    print(f"Probando combinación {i}/{len(candidatos)}: {params}")
    ndcg_val = evaluar_ndcg_en_validacion(params)
    print(f"nDCG@10 validación = {ndcg_val:.4f}")

    if ndcg_val > mejor_ndcg:
        mejor_ndcg = ndcg_val
        mejores_params = params
        print(f"Nuevo mejor modelo con nDCG@10 = {mejor_ndcg:.4f}")

print("Mejores hiperparámetros según nDCG@10 en validación:")
print(mejores_params)
print(f"nDCG@10 validación = {mejor_ndcg:.4f}")




ranker = XGBRanker(
    objective="rank:pairwise",
    random_state=42,
    tree_method="hist",
    n_jobs=-1,
    **mejores_params
)

ranker.fit(
    X_train,
    y_train,
    group=group_train
)

y_pred = ranker.predict(X_test)

recomendaciones = defaultdict(list)
for i in range(X_test_df.shape[0]):
    user_id = X_test_df.iloc[i]['user_id']
    app_id = X_test_df.iloc[i]['app_id']
    predicted_score = y_pred[i]
    recomendaciones[user_id].append([app_id, predicted_score])

recomendaciones_def = defaultdict(list)
for usuario, lista_recomendaciones in recomendaciones.items():
    recomendaciones_ord = sorted(lista_recomendaciones, key=lambda x: x[1], reverse=True)
    recomendaciones_ord = [i[0] for i in recomendaciones_ord]
    recomendaciones_def[usuario] = recomendaciones_ord

precision_list = []
recall_list = []
ndcg_list = []
f1_list = []
hitrate_list = []
map10_list = []
diversity_list = []

for usuario, recomendaciones_usuario in recomendaciones_def.items():
    items_rel_usuario = items_relevantes_test.get(usuario, [])
    if len(items_rel_usuario) == 0:
        continue

    recomendaciones_10 = recomendaciones_usuario[:10]

    precision_usuario = precision_at_k(recomendaciones_10, items_rel_usuario)
    recall_usuario = recall_at_k(recomendaciones_10, items_rel_usuario)
    ndcg_usuario = ndcg_at_k(recomendaciones_10, items_rel_usuario)
    f1_usuario = f1_at_k(recomendaciones_10, items_rel_usuario)
    hitrate_usuario = hit_score_at_k(recomendaciones_10, items_rel_usuario)
    map10_usuario = map_at_k(recomendaciones_10, items_rel_usuario)
    diversity_usuario = diversity_at_k(recomendaciones_10, info_videojuegos)

    precision_list.append(precision_usuario)
    recall_list.append(recall_usuario)
    ndcg_list.append(ndcg_usuario)
    f1_list.append(f1_usuario)
    hitrate_list.append(hitrate_usuario)
    map10_list.append(map10_usuario)
    diversity_list.append(diversity_usuario)

print(f"Precision@10: {np.mean(precision_list):.4f}")
print(f"Recall@10: {np.mean(recall_list):.4f}")
print(f"nDCG@10: {np.mean(ndcg_list):.4f}")
print(f"F1-Score@10: {np.mean(f1_list):.4f}")
print(f"Hit Score@10: {np.mean(hitrate_list):.4f}")
print(f"MAP@10: {np.mean(map10_list):.4f}")
print(f"Diversity: {np.mean(diversity_list):.4f}")
