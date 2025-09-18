import warnings
warnings.filterwarnings("ignore")

import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import __version__ as skver
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    RocCurveDisplay, PrecisionRecallDisplay, precision_recall_curve,
    balanced_accuracy_score, matthews_corrcoef, ConfusionMatrixDisplay,
    roc_curve, auc
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import EasyEnsembleClassifier

# ========= CONFIG =========
FAST = True              # True = rápido para iterar, False = completo
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_ROWS_FAST = 250_000  # si el CSV es muy grande, reducimos para FAST
TARGET_RECALL = 0.80     # meta de recall (clase 1)
# =========================

# Carpeta de salida con timestamp para no sobreescribir
RUN_ID = time.strftime("%Y%m%d-%H%M%S")
OUTDIR = f"outputs_{RUN_ID}"
os.makedirs(OUTDIR, exist_ok=True)

def save_and_show(filename):
    """Guarda la figura actual en OUTDIR y hace show()."""
    path = os.path.join(OUTDIR, filename)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {path}")

def pick_target_column(df):
    if "fraud_bool" in df.columns: return "fraud_bool"
    if "fraud-bool" in df.columns: return "fraud-bool"
    raise ValueError("No encuentro la columna objetivo: 'fraud_bool'/'fraud-bool'.")

def build_preprocessor(X):
    """ColumnTransformer con StandardScaler (num) y OneHotEncoder (cat)."""
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    print(f"[Preproc] Numéricas: {len(num_cols)} | Categóricas: {len(cat_cols)}")

    major, minor = map(int, skver.split(".")[:2])
    if (major, minor) >= (1, 2):
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        scaler = StandardScaler(with_mean=False)  # compatible con matrices dispersas
    else:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
        scaler = StandardScaler(with_mean=False)

    numeric_transformer = Pipeline(steps=[("scaler", scaler)])
    categorical_transformer = Pipeline(steps=[("encoder", ohe)])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ],
        remainder="drop",
        sparse_threshold=0.3
    )
    return preprocessor

def build_models(fast=True):
    """Devuelve (ensemble_soft, eec)."""
    rf = RandomForestClassifier(
        n_estimators=80 if fast else 350,
        max_depth=None,
        min_samples_leaf=2 if fast else 1,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE
    )

    gb = GradientBoostingClassifier(
        n_estimators=100 if fast else 300,
        learning_rate=0.10 if fast else 0.08,
        subsample=1.0,
        random_state=RANDOM_STATE
    )

    lr = LogisticRegression(max_iter=1000, class_weight="balanced")

    ensemble_soft = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
        voting="soft"
    )

    eec = EasyEnsembleClassifier(
        n_estimators=10 if fast else 30,
        random_state=RANDOM_STATE
    )
    return ensemble_soft, eec

def choose_threshold_for_target_recall(y_true, y_prob, target_recall=0.80):
    """Umbral con mejor precisión sujeto a recall >= target_recall.
       Si no existe, toma el de mayor recall posible.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    thresholds = np.r_[0.0, thresholds]  # alinear longitudes

    mask = recalls >= target_recall
    if mask.any():
        cand_idx = np.where(mask)[0]
        best_local = cand_idx[np.argmax(precisions[mask])]
        thr = thresholds[best_local]
    else:
        best_local = np.argmax(recalls)
        thr = thresholds[best_local]

    print(f"[Thr@Recall] Objetivo recall≥{target_recall:.2f} → "
          f"thr={thr:.4f} | recall={recalls[best_local]:.3f} | precision={precisions[best_local]:.3f}")
    return float(thr), (precisions, recalls, thresholds)

def plot_confusion_matrix_counts(cm, labels=("No Fraude", "Fraude"), title="Matriz de Confusión"):
    plt.figure()
    plt.title(title)
    plt.imshow(cm, interpolation="nearest")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, int(v), ha="center", va="center")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    save_and_show("confusion_counts.png")

def main():
    print("=== 1) Cargando CSV ===")
    df = pd.read_csv("Base.csv")

    target_col = pick_target_column(df)
    print(f"[Info] Target: {target_col}")

    if FAST and len(df) > MAX_ROWS_FAST:
        print(f"[FAST] Submuestreando filas para prueba rápida (máx {MAX_ROWS_FAST:,})…")
        pos = df[df[target_col] == 1]
        neg = df[df[target_col] == 0]
        need_neg = MAX_ROWS_FAST - len(pos)
        need_neg = max(need_neg, int(0.8 * MAX_ROWS_FAST))
        neg_sample = neg.sample(n=min(len(neg), max(1, need_neg)), random_state=RANDOM_STATE)
        df = pd.concat([pos, neg_sample]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"[FAST] Nuevo tamaño: {len(df):,} | Positivos: {df[target_col].sum():,}")

    print("=== 2) Separando X/y ===")
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    print("=== 3) Train/Test Split (estratificado) ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"[Split] Train: {X_train.shape}  Test: {X_test.shape}")
    print(f"[Distrib] Train fraud={y_train.mean():.4f} | Test fraud={y_test.mean():.4f}")

    print("=== 4) Construyendo preprocesadores ===")
    pre_soft = build_preprocessor(X_train)
    pre_eec  = build_preprocessor(X_train)

    print("=== 5) Configurando modelos (ensemble + EEC) ===")
    ensemble_soft, eec = build_models(fast=FAST)

    print("=== 6) Creando pipelines ===")
    if FAST:
        balancer = RandomUnderSampler(sampling_strategy=0.5, random_state=RANDOM_STATE)
        print("[Balanceo] RandomUnderSampler(sampling_strategy=0.5)")
    else:
        balancer = SMOTE(sampling_strategy=1.0, random_state=RANDOM_STATE, k_neighbors=5)
        print("[Balanceo] SMOTE(sampling_strategy=1.0)")

    pipe_soft = ImbPipeline(steps=[
        ("pre", pre_soft),
        ("bal", balancer),
        ("clf", ensemble_soft)
    ])

    eec_pipe = Pipeline(steps=[
        ("pre", pre_eec),
        ("clf", eec)
    ])

    print("=== 7) Entrenando pipelines (puede tardar) ===")
    print("[Fit] Voting Soft…")
    pipe_soft.fit(X_train, y_train)
    print("[Fit] EasyEnsemble…")
    eec_pipe.fit(X_train, y_train)

    print("=== 8) Predicción de probabilidades y ajuste de umbral a recall objetivo ===")
    proba_soft = pipe_soft.predict_proba(X_test)[:, 1]
    proba_eec  = eec_pipe.predict_proba(X_test)[:, 1]
    # Promedio simple (puedes ponderar, p.ej. 0.6*soft + 0.4*eec)
    y_prob = (proba_soft + proba_eec) / 2.0

    thr, (precisions, recalls, thresholds_pr_aligned) = choose_threshold_for_target_recall(
        y_test, y_prob, target_recall=TARGET_RECALL
    )
    y_pred = (y_prob >= thr).astype(int)

    print("=== 9) Métricas ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    print("=== 10) Gráficos base ===")
    # 10.1 Matriz de confusión (conteos) → guarda dentro de la función
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix_counts(cm, labels=("No Fraude", "Fraude"), title="Matriz de Confusión")

    # 10.2 Curva ROC
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("Curva ROC")
    save_and_show("roc_curve.png")

    # 10.3 Curva Precision-Recall
    PrecisionRecallDisplay.from_predictions(y_test, y_prob)
    plt.title("Curva Precision-Recall")
    save_and_show("pr_curve.png")

    print("=== 10B) Gráficas extendidas para reporte ===")

    # 10B.1 Matriz de confusión normalizada (%)
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, normalize='true',
        display_labels=["No Fraude", "Fraude"]
    )
    plt.title("Matriz de Confusión Normalizada (%)")
    plt.tight_layout()
    save_and_show("confusion_normalized.png")

    # 10B.2 Distribución de probabilidades por clase
    probs_0 = y_prob[y_test == 0]
    probs_1 = y_prob[y_test == 1]
    bins = 50
    plt.figure()
    plt.hist(probs_0, bins=bins, density=True, alpha=0.5, label="Clase 0 (No Fraude)")
    plt.hist(probs_1, bins=bins, density=True, alpha=0.5, label="Clase 1 (Fraude)")
    plt.axvline(thr, linestyle='--', linewidth=2, label=f"Umbral = {thr:.3f}")
    plt.xlabel("Probabilidad predicha de fraude")
    plt.ylabel("Densidad")
    plt.title("Distribución de Probabilidades por Clase")
    plt.legend()
    plt.tight_layout()
    save_and_show("score_distribution.png")

    # 10B.3 Precision y Recall vs Umbral (alineado correctamente)
    precisions_pr, recalls_pr, thresholds_pr = precision_recall_curve(y_test, y_prob)
    plt.figure()
    # thresholds_pr: longitud m; precisions/recalls: m+1 → usamos [: -1]
    plt.plot(thresholds_pr, precisions_pr[:-1], label="Precision")
    plt.plot(thresholds_pr, recalls_pr[:-1],   label="Recall")
    plt.axvline(thr, linestyle='--', linewidth=1, label=f"Umbral usado = {thr:.3f}")
    plt.xlabel("Umbral")
    plt.ylabel("Métrica")
    plt.title("Precision y Recall vs Umbral")
    plt.legend()
    plt.tight_layout()
    save_and_show("precision_recall_vs_threshold.png")

    # 10B.4 KS Curve (muy usada en banca)
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_prob)
    ks_statistic = np.max(tpr - fpr)
    plt.figure()
    plt.plot(thresholds_roc, tpr - fpr)
    plt.axvline(thr, linestyle='--', linewidth=1, label=f"Umbral usado = {thr:.3f}")
    plt.title(f"KS Curve (KS = {ks_statistic:.3f})")
    plt.xlabel("Umbral")
    plt.ylabel("TPR - FPR")
    plt.legend()
    plt.tight_layout()
    save_and_show("ks_curve.png")

    # === 11) Métricas adicionales (para el reporte) ===
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    pr_auc_val = auc(recalls_pr, precisions_pr)  # área bajo la curva PR

    print("=== Métricas adicionales ===")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"PR AUC: {pr_auc_val:.4f}")

    # Guardar métricas a disco
    summary_txt = os.path.join(OUTDIR, "metrics_summary.txt")
    with open(summary_txt, "w") as f:
        f.write(f"Umbral: {thr:.4f}\n")
        f.write(f"Recall objetivo: {TARGET_RECALL:.2f}\n")
        f.write(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}\n")
        f.write(f"PR AUC: {pr_auc_val:.4f}\n")
        f.write(f"Balanced Accuracy: {bal_acc:.4f}\n")
        f.write(f"MCC: {mcc:.4f}\n")
    print(f"[Saved] {summary_txt}")

    print("=== Listo ===")
    print(f"[Resumen] Umbral usado: {thr:.4f} | Recall objetivo: {TARGET_RECALL:.2f} | FAST={FAST}")
    print(f"[Output dir] {OUTDIR}")

if __name__ == "__main__":
    main()