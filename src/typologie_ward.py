import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]   
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams["axes.grid"] = True

df = pd.read_excel(DATA_DIR / "DANE.xlsx")

df_clean = df.copy()
df.head()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread(DATA_DIR / "cechy.png")

plt.imshow(img)
plt.axis("off")
plt.show()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread(DATA_DIR / "proba badawcza.png")


plt.imshow(img)
plt.axis("off")
plt.show()

description_cols = ["adress", "type", "name", "context","lat","lon"]

for col in description_cols:
    cat_col = col + "_cat"
    if cat_col in df_clean.columns:
        df_clean.drop(columns=[cat_col], inplace=True)

derived_cols = [
    "plane_area",
    "proportions_facade_angle",
    "proportions_facade_body_angle",
    "horizontal_roof_area",
    "horizontal_top_area",
    "horizontal_upper_area",
    "horizontal_ground_area",
    "belt_top",
    "belt_top_area",
    "belt_groundscape_height_percent",
    "belt_groundscape_length_percent",
    "vertical_area_percent",
    "openings_perforation",
    "windows_perforation_ground",
    "windows_perforation_upper",
    "Lat",
    "Lon",
]

df_clean = df.drop(columns=[c for c in derived_cols if c in df.columns])

binary_map = {
    "tak": 1, "nie": 0,
    "Tak": 1, "Nie": 0,
    "TAK": 1, "NIE": 0,
    True: 1, False: 0
}

binary_cols = []

object_cols = [
    c for c in df_clean.select_dtypes(include=["object"]).columns
    if c not in description_cols
]

for col in object_cols:
    unique_vals = set(df_clean[col].dropna().unique())
    if len(unique_vals) > 0 and unique_vals.issubset(binary_map.keys()):
        df_clean[col] = df_clean[col].map(binary_map)
        binary_cols.append(col)

categorical_cols = []

for col in df_clean.select_dtypes(include=["object"]).columns:
    if col not in binary_cols:
        if df_clean[col].nunique(dropna=True) > 1:
            categorical_cols.append(col)
            df_clean[col + "_cat"] = df_clean[col].astype("category").cat.codes

numeric_cols = df_clean.select_dtypes(include=["number"]).columns.tolist()
X_num = df_clean[numeric_cols]

low_var_cols = X_num.var()[X_num.var() < 0.01].index.tolist()
X_num = X_num.drop(columns=low_var_cols)
numeric_cols = X_num.columns.tolist()

corr = X_num.corr()

plt.figure(figsize=(14,12))
plt.imshow(corr, vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
plt.yticks(range(len(numeric_cols)), numeric_cols)
plt.tight_layout()
plt.show()

high_corr_pairs = []
for i, c1 in enumerate(numeric_cols):
    for j, c2 in enumerate(numeric_cols):
        if j <= i: 
            continue
        if abs(corr.loc[c1, c2]) > 0.85:
            high_corr_pairs.append((c1, c2, corr.loc[c1, c2]))

high_corr_pairs

cols_to_remove_after_corr = [
    "belt_top_length",
    "belt_groundscape_length",
    "proportions_facade_body",
    "lowest_window_row_avarage_height_m",
    "belt_groundscape_area",
    "vertical_area",
    "vertical_count",
    "layering_out_area",
    "layering_out_length",
    "symm opp upper",
    "adress_cat",
    "type_cad",
    "name_cat",
    "context_cat",
]

cols_to_remove_after_corr = [c for c in cols_to_remove_after_corr if c in numeric_cols]

feature_cols_final = [c for c in numeric_cols if c not in cols_to_remove_after_corr]

X_final = df_clean[feature_cols_final]

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X_final)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_scaled)

df_clean["cluster"] = labels
df_clean[["cluster"]].head()
df_clean["cluster"].value_counts()

import optuna
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def objective(trial):
    n_clusters = trial.suggest_int("n_clusters", 2, 10)
    n_init = trial.suggest_int("n_init", 10, 50)
    max_iter = trial.suggest_int("max_iter", 100, 500)

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=42
    )
    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, n_jobs=-1)

best_params = study.best_params
best_score = study.best_value

print("Najlepsze parametry:", best_params)
print("Najlepszy silhouette_score:", best_score)

from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

linked = linkage(X_scaled, method="ward")
labels = df_clean["adress"].values

plt.figure(figsize=(18, 8))

plt.title("Dendrogram fasad – metoda Warda", fontsize=13, pad=10)
plt.ylabel("Odległość (Ward)", fontsize=12)

threshold = 0.7 * linked[:, 2].max()

dend = dendrogram(
    linked,
    labels=labels,
    leaf_rotation=90,
    leaf_font_size=9,
    color_threshold=threshold,      
    above_threshold_color="lightgrey"  
)

ax = plt.gca()
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from tqdm import tqdm      
import matplotlib.pyplot as plt


X = X_scaled          
k = 7               
B = 100          

n = X.shape[0]
co_matrix = np.zeros((n, n))

for b in tqdm(range(B), desc="Bootstrap Ward"):
    idx = np.random.choice(np.arange(n), size=n, replace=True)
    X_boot = X[idx]

    Z_boot = linkage(X_boot, method='ward')
    labels_boot = fcluster(Z_boot, k, criterion='maxclust')

    for i_new, i_old in enumerate(idx):
        for j_new, j_old in enumerate(idx):
            if labels_boot[i_new] == labels_boot[j_new]:
                co_matrix[i_old, j_old] += 1

stability = co_matrix / B

Z_full = linkage(X, method='ward')
labels_full = fcluster(Z_full, k, criterion='maxclust')

cluster_stability = []
for cluster_id in range(1, k + 1):
    idx_cluster = np.where(labels_full == cluster_id)[0]
    sub = stability[np.ix_(idx_cluster, idx_cluster)]
    cluster_stability.append(sub.mean())

print("Stabilność klastrów:")
for i, s in enumerate(cluster_stability, start=1):
    print(f"  Klaster {i}: {s:.3f}")

    from scipy.cluster.hierarchy import fcluster
import numpy as np

clusters = fcluster(linked, t=15, criterion='distance')

unique_clusters = np.unique(clusters)
print("Liczba klastrów:", len(unique_clusters))
print("Numery klastrów:", unique_clusters)

df_clean["cluster"] = clusters
df_clean[["adress", "cluster"]].head(60)

counts = df_clean["cluster"].value_counts().sort_index()
plt.figure(figsize=(6,4))
plt.bar(counts.index, counts.values)
plt.xlabel("Klaster")
plt.ylabel("Liczba fasad")
plt.title("Liczebność klastrów (Ward)")
plt.show()

cluster_profile = df_clean.groupby("cluster")[feature_cols_final].mean()
cluster_profile

df_scaled = pd.DataFrame(
    X_scaled,
    columns=feature_cols_final
)

df_scaled["cluster"] = df_clean["cluster"].values

cluster_profile_scaled = df_scaled.groupby("cluster").mean()
cluster_profile_scaled

import seaborn as sns
plt.figure(figsize=(15,6))
sns.heatmap(cluster_profile_scaled, cmap="viridis")
plt.title("Profil cech w 6 klastrach fasad (Ward)")
plt.show()

from sklearn.decomposition import PCA
import plotly.express as px

palette = ["#e41a1c", "#377eb8", "#4daf4a",
           "#984ea3", "#ff7f00", "#ffff33"]

cluster_ids = sorted(df_clean["cluster"].unique())

cluster_labels = {
    1: "Klasa 1 - fasady z czescia dachowa i dominantami",
    2: "Klasa 2 - fasady monumentalne, z wykuszami , wysokimi gzysmami, o umiarkowanej perforacji",
    3: "Klasa 3 - fasady pionowe, wysokie z czescia dachowa o umiarkowanej perforacji, z niesymetrycznymi wykuszami",
    4: "Klasa 4 - fasady z ryzalitami i czescia dachowa z oknami, niesymetryczne",
    5: "Klasa 5 - fasady niskie, bez czesci dachowej, proste",
    6: "Klasa 6 - fasady dlugie z ryzalitami z niskim gzymsem przyziemia, dosc symetryczne",
    7: "Klasa 7 - fasady o sredniej wysokosci, zazwyczaj kwadratowe i wysokim przyziemiu tzw. sredniaki/typowe"
}

for cl in cluster_ids:
    if cl not in cluster_labels:
        cluster_labels[cl] = f"Klaster {cl}"

df_clean["cluster_label"] = df_clean["cluster"].map(cluster_labels)

color_map = {
    cluster_labels[cl]: palette[i % len(palette)]
    for i, cl in enumerate(cluster_ids)
}

pca3 = PCA(n_components=3)
coords3 = pca3.fit_transform(X_scaled)

df_clean["pca1"] = coords3[:, 0]
df_clean["pca2"] = coords3[:, 1]
df_clean["pca3"] = coords3[:, 2]

fig = px.scatter_3d(
    df_clean,
    x="pca1",
    y="pca2",
    z="pca3",
    color="cluster_label",
    color_discrete_map=color_map,
    hover_data=["adress", "type", "context", "cluster"]
)

fig.update_layout(
    scene=dict(
        xaxis_title="PCA 1",
        yaxis_title="PCA 2",
        zaxis_title="PCA 3"
    ),
    title="PCA 3D – wizualizacja klastrów",
    legend_title="Typ klastra"
)

fig.show()

import folium

Lat = df["Lat"]
Lon = df["Lon"]

df["cluster"] = df_clean["cluster"].values

m = folium.Map(
    location=[Lat.mean(), Lon.mean()],
    zoom_start=13,
    tiles="CartoDB.VoyagerNoLabels"
)

cluster_ids = sorted(df["cluster"].unique())
palette = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#ffff33"]
color_map = {cl: palette[i % len(palette)] for i, cl in enumerate(cluster_ids)}

# --- rysowanie punktów ---
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["Lat"], row["Lon"]],
        radius=5,
        weight=1,
        color=color_map[row["cluster"]],
        fill=True,
        fill_color=color_map[row["cluster"]],
        fill_opacity=0.9,
        popup=(
            f"Adres: {row.get('adress','brak')}<br>"
            f"Typ: {row.get('type','brak')}<br>"
            f"Kontekst: {row.get('context','brak')}<br>"
            f"Klastr: {row['cluster']}"
        )
    ).add_to(m)

m.save(OUTPUT_DIR / "mapa_klastry_ward.html")
