"""
segmentation.py
----------------
Core pipeline for Project 3: Unsupervised Learning (Customer Segmentation).

Implements the four-stage Input-Process-Output architecture from the brief:
    1. SCALE      -> StandardScaler
    2. COMPRESS   -> PCA (dimensionality reduction, 95% variance retained)
    3. CLUSTER    -> K-Means, with K chosen via Elbow Method + Silhouette Score
    4. TRANSLATE  -> Inverse-transform centroids back to human-readable
                      metrics and build business personas

This module can be imported by the notebook, or run directly:
    python src/segmentation.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "customer_data.csv"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42

# Columns that are identifiers/categorical and should NOT go into the
# distance-based clustering math directly.
NON_FEATURE_COLS = ["customer_id", "gender", "region", "preferred_channel"]


# ---------------------------------------------------------------------------
# Stage 0: Load + clean
# ---------------------------------------------------------------------------
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Impute the small amount of injected missingness with the median —
    # simple and defensible for a handful of numeric NaNs.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric behavioral/demographic columns used for clustering."""
    return df.drop(columns=NON_FEATURE_COLS)


# ---------------------------------------------------------------------------
# Stage 1: SCALE
# ---------------------------------------------------------------------------
def scale_features(X: pd.DataFrame):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# ---------------------------------------------------------------------------
# Stage 2: COMPRESS (PCA)
# ---------------------------------------------------------------------------
def run_pca(X_scaled: np.ndarray, variance_threshold: float = 0.95):
    """Fit PCA with all components first to inspect the variance curve,
    then refit keeping only enough components to reach `variance_threshold`.
    """
    pca_full = PCA(random_state=RANDOM_SEED).fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(cum_var >= variance_threshold) + 1)
    n_components = max(n_components, 2)  # keep at least 2D for plotting

    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca, pca_full, cum_var, n_components


def plot_explained_variance(cum_var: np.ndarray, chosen_k: int):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, len(cum_var) + 1)
    ax.plot(x, cum_var, marker="o", color="#1f3864")
    ax.axhline(0.95, color="#d4a017", linestyle="--", label="95% threshold")
    ax.axvline(chosen_k, color="#c0392b", linestyle=":", label=f"Chosen k={chosen_k}")
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA: Cumulative Explained Variance")
    ax.set_xticks(x)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pca_explained_variance.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stage 3: CLUSTER (K-Means, tuned via Elbow + Silhouette)
# ---------------------------------------------------------------------------
def evaluate_k_range(X_pca: np.ndarray, k_range=range(2, 11)):
    wcss = []
    sil_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X_pca)
        wcss.append(km.inertia_)
        sil_scores.append(silhouette_score(X_pca, labels))
    return list(k_range), wcss, sil_scores


def find_elbow_k(k_values, wcss):
    """Locate the elbow via the 'kneedle' knee-point heuristic; falls back
    to a simple max-curvature calculation if the `kneed` package is absent.
    """
    try:
        from kneed import KneeLocator

        kl = KneeLocator(k_values, wcss, curve="convex", direction="decreasing")
        if kl.knee is not None:
            return int(kl.knee)
    except Exception:
        pass

    # Fallback: point of maximum distance from the line connecting endpoints
    k_arr = np.array(k_values, dtype=float)
    w_arr = np.array(wcss, dtype=float)
    w_norm = (w_arr - w_arr.min()) / (w_arr.max() - w_arr.min())
    k_norm = (k_arr - k_arr.min()) / (k_arr.max() - k_arr.min())
    p1 = np.array([k_norm[0], w_norm[0]])
    p2 = np.array([k_norm[-1], w_norm[-1]])
    line_vec = p2 - p1
    line_vec /= np.linalg.norm(line_vec)
    distances = []
    for kk, ww in zip(k_norm, w_norm):
        p = np.array([kk, ww])
        proj_len = np.dot(p - p1, line_vec)
        proj_point = p1 + proj_len * line_vec
        distances.append(np.linalg.norm(p - proj_point))
    return int(k_values[int(np.argmax(distances))])


def plot_elbow(k_values, wcss, chosen_k):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, wcss, marker="o", color="#1f3864")
    ax.axvline(chosen_k, color="#c0392b", linestyle="--", label=f"Elbow k={chosen_k}")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Within-Cluster Sum of Squares (WCSS)")
    ax.set_title("Elbow Method")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "elbow_method.png", dpi=150)
    plt.close(fig)


def plot_silhouette(k_values, sil_scores, chosen_k):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, sil_scores, marker="o", color="#8b1e1e")
    best_k = k_values[int(np.argmax(sil_scores))]
    ax.axvline(chosen_k, color="#1f3864", linestyle="--", label=f"Chosen k={chosen_k}")
    ax.axvline(best_k, color="#2e7d32", linestyle=":", label=f"Best silhouette k={best_k}")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score by K")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "silhouette_scores.png", dpi=150)
    plt.close(fig)
    return best_k


def choose_final_k(k_values, wcss, sil_scores) -> int:
    """Combine both diagnostics: prefer the silhouette-best k, but if the
    elbow and silhouette disagree by only 1, and the elbow k has a
    competitive silhouette score, keep the elbow k (favors business-usable,
    slightly simpler segmentations, per the brief's persona-translation
    requirement)."""
    elbow_k = find_elbow_k(k_values, wcss)
    sil_best_k = k_values[int(np.argmax(sil_scores))]
    if elbow_k == sil_best_k:
        return elbow_k
    elbow_idx = k_values.index(elbow_k)
    sil_best_score = max(sil_scores)
    elbow_score = sil_scores[elbow_idx]
    if elbow_score >= 0.9 * sil_best_score:
        return elbow_k
    return sil_best_k


def fit_final_kmeans(X_pca: np.ndarray, k: int):
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(X_pca)
    return km, labels


def plot_clusters_2d(X_pca: np.ndarray, labels: np.ndarray, k: int):
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("Set1", k)
    for cluster_id in range(k):
        mask = labels == cluster_id
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            s=25, alpha=0.7, color=palette[cluster_id], label=f"Cluster {cluster_id}",
        )
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title(f"Customer Segments in PCA Space (K={k})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cluster_scatter_pca.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stage 4: TRANSLATE (business personas)
# ---------------------------------------------------------------------------
def build_persona_table(df_features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Average the ORIGINAL (unscaled) feature values per cluster — this is
    the human-readable translation step, distinct from the PCA-space
    centroids used internally by K-Means."""
    tmp = df_features.copy()
    tmp["cluster"] = labels
    persona = tmp.groupby("cluster").mean(numeric_only=True).round(2)
    persona["customer_count"] = tmp.groupby("cluster").size()
    persona["pct_of_base"] = (persona["customer_count"] / len(tmp) * 100).round(1)
    return persona


def label_personas(persona: pd.DataFrame) -> dict:
    """Simple rule-based naming from income/spending_score quadrant —
    mirrors the four-quadrant persona matrix in the source brief."""
    income_median = persona["income"].median()
    score_median = persona["spending_score"].median()
    names = {}
    for cluster_id, row in persona.iterrows():
        high_income = row["income"] >= income_median
        high_spend = row["spending_score"] >= score_median
        if high_income and not high_spend:
            names[cluster_id] = "Affluent Conservatives"
        elif high_income and high_spend:
            names[cluster_id] = "High-Value Trendsetters"
        elif not high_income and high_spend:
            names[cluster_id] = "Budget-Conscious Explorers"
        else:
            names[cluster_id] = "Conservative Minimizers"
    return names


PERSONA_ACTIONS = {
    "Affluent Conservatives": "High-touch support, extended warranties, loyalty programs.",
    "High-Value Trendsetters": "Exclusive perks, early access, experiential marketing.",
    "Budget-Conscious Explorers": "Influencer campaigns, flash sales, buy-now-pay-later.",
    "Conservative Minimizers": "Minimize acquisition spend, clear price/value messaging, basic utility offers.",
}


# ---------------------------------------------------------------------------
# End-to-end run
# ---------------------------------------------------------------------------
def run_pipeline():
    df = load_data()
    X = get_feature_matrix(df)

    X_scaled, scaler = scale_features(X)
    X_pca, pca, pca_full, cum_var, n_components = run_pca(X_scaled)
    plot_explained_variance(cum_var, n_components)

    k_values, wcss, sil_scores = evaluate_k_range(X_pca)
    plot_elbow(k_values, wcss, find_elbow_k(k_values, wcss))
    plot_silhouette(k_values, sil_scores, choose_final_k(k_values, wcss, sil_scores))

    final_k = choose_final_k(k_values, wcss, sil_scores)
    km, labels = fit_final_kmeans(X_pca, final_k)
    plot_clusters_2d(X_pca, labels, final_k)

    df["cluster"] = labels
    persona = build_persona_table(X, labels)
    persona_names = label_personas(persona)
    persona["persona_name"] = persona.index.map(persona_names)
    persona["recommended_action"] = persona["persona_name"].map(PERSONA_ACTIONS)

    df["persona_name"] = df["cluster"].map(persona_names)

    df.to_csv(OUT_DIR / "customers_with_segments.csv", index=False)
    persona.to_csv(OUT_DIR / "persona_summary.csv")

    print(f"PCA components retained (95% variance): {n_components}")
    print(f"Final K chosen: {final_k}")
    print("\nPersona summary:")
    print(persona[["customer_count", "pct_of_base", "persona_name"]])

    return {
        "df": df,
        "persona": persona,
        "final_k": final_k,
        "n_components": n_components,
        "silhouette": max(sil_scores),
    }


if __name__ == "__main__":
    run_pipeline()
