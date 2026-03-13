# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import warnings
import re
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, roc_curve

warnings.filterwarnings("ignore")

cli_train_path = r"C:\Users\zength\Desktop\cli_train.csv"
cli_test1_path = r"C:\Users\zength\Desktop\cli_test1.csv"
cli_test2_path = r"C:\Users\zength\Desktop\cli_test2.csv"

clinical_features = ['治疗方案', '年龄', 'T', 'N', '组织学分级', 'Ki67']
label_col = '整体1=PCR'

def load_cli(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[clinical_features + [label_col]].copy()
    return df

def to_stage_cat(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    m = re.search(r'(\d+)', s)
    return m.group(1) if m else s

def prep_X(df):
    X = df[clinical_features].copy()
    for c in ['T', 'N']:
        X[c] = X[c].apply(to_stage_cat)
    for c in ['年龄', 'Ki67']:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    return X

def _safe_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp

def bootstrap_metrics_ci(y_true, y_prob, threshold=0.5, n_boot=2000, seed=42):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = _safe_confusion(y_true, y_pred)

    auc_pt = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    acc_pt = accuracy_score(y_true, y_pred)
    sen_pt = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spe_pt = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv_pt = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv_pt = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs, accs, sens, spes, ppvs, npvs = [], [], [], [], [], []

    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt = y_true[idx]
        ys = y_prob[idx]
        yp = (ys >= threshold).astype(int)

        if len(np.unique(yt)) == 2:
            aucs.append(roc_auc_score(yt, ys))
        accs.append(accuracy_score(yt, yp))
        tn_b, fp_b, fn_b, tp_b = _safe_confusion(yt, yp)
        sens.append(tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else np.nan)
        spes.append(tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else np.nan)
        ppvs.append(tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else np.nan)
        npvs.append(tn_b / (tn_b + fn_b) if (tn_b + fn_b) > 0 else np.nan)

    def ci(arr):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return (np.nan, np.nan)
        return (np.percentile(arr, 2.5), np.percentile(arr, 97.5))

    return {
        "AUC": (auc_pt, *ci(aucs)),
        "ACC": (acc_pt, *ci(accs)),
        "Se":  (sen_pt, *ci(sens)),
        "Sp":  (spe_pt, *ci(spes)),
        "PPV": (ppv_pt, *ci(ppvs)),
        "NPV": (npv_pt, *ci(npvs)),
        "threshold": threshold
    }

def print_metrics_with_ci(title, d):
    print(f"\n=== {title} ===")
    print(f"Threshold: {d['threshold']:.4f}")
    for k in ["AUC", "ACC", "Se", "Sp", "PPV", "NPV"]:
        pt, lo, hi = d[k]
        print(f"{k}: {pt:.3f} (95% CI: {lo:.3f}–{hi:.3f})")

cli_train = load_cli(cli_train_path)
cli_test1 = load_cli(cli_test1_path)
cli_test2 = load_cli(cli_test2_path)

X_train = prep_X(cli_train)
y_train = cli_train[label_col].astype(int).values
X_test1 = prep_X(cli_test1)
y_test1 = cli_test1[label_col].astype(int).values
X_test2 = prep_X(cli_test2)
y_test2 = cli_test2[label_col].astype(int).values

numeric_features = ['年龄', 'Ki67']
categorical_features = ['治疗方案', 'T', 'N', '组织学分级']

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

pipe = Pipeline([
    ("preprocess", preprocess),
    ("clf", LogisticRegression(max_iter=5000, solver="saga"))
])

param_grid = {
    "clf__penalty": ["l2", "l1", "elasticnet"],
    "clf__C": np.logspace(-2, 1, 100),
    "clf__l1_ratio": [0.2, 0.5, 0.8],
    "clf__class_weight": [None, "balanced"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)

gs = GridSearchCV(
    pipe, param_grid=param_grid, scoring="roc_auc",
    cv=cv, n_jobs=-1, refit=True
)
gs.fit(X_train, y_train)

best_model = gs.best_estimator_
print("\n=== Best hyperparameters (Training CV) ===")
print(gs.best_params_)
print(f"Best CV AUC: {gs.best_score_:.3f}")

oof_prob = cross_val_predict(
    best_model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
)[:, 1]
fpr, tpr, thr = roc_curve(y_train, oof_prob)
best_thr = float(thr[np.argmax(tpr - fpr)])
print(f"Chosen threshold (Youden J, training OOF): {best_thr:.4f}")

train_prob = best_model.predict_proba(X_train)[:, 1]
test1_prob = best_model.predict_proba(X_test1)[:, 1]
test2_prob = best_model.predict_proba(X_test2)[:, 1]

train_metrics = bootstrap_metrics_ci(y_train, train_prob, threshold=best_thr, n_boot=2000, seed=23)
test1_metrics = bootstrap_metrics_ci(y_test1, test1_prob, threshold=best_thr, n_boot=2000, seed=23)
test2_metrics = bootstrap_metrics_ci(y_test2, test2_prob, threshold=best_thr, n_boot=2000, seed=23)

print_metrics_with_ci("Clinical model (Train)", train_metrics)
print("Confusion matrix (Train):\n", confusion_matrix(y_train, (train_prob >= best_thr).astype(int)))

print_metrics_with_ci("Clinical model (Test1)", test1_metrics)
print("Confusion matrix (Test1):\n", confusion_matrix(y_test1, (test1_prob >= best_thr).astype(int)))

print_metrics_with_ci("Clinical model (Test2)", test2_metrics)
print("Confusion matrix (Test2):\n", confusion_matrix(y_test2, (test2_prob >= best_thr).astype(int)))
