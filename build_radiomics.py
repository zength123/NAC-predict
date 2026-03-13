# -*- coding: UTF-8 -*-
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import itertools
import warnings
import joblib

warnings.filterwarnings('ignore')

def bootstrap_ci(y_true, y_pred, y_pred_proba, n_bootstrap=2000, random_seed=42):
    np.random.seed(random_seed)
    n = len(y_true)
    accs, aucs, f1s, sens, specs = [], [], [], [], []
    for _ in range(n_bootstrap):
        idx = np.random.choice(np.arange(n), size=n, replace=True)
        yt = y_true[idx]
        yp = y_pred[idx]
        yp_proba = y_pred_proba[idx]
        try:
            auc_val = roc_auc_score(yt, yp_proba)
        except:
            auc_val = np.nan
        accs.append(accuracy_score(yt, yp))
        aucs.append(auc_val)
        p, r, f1, _ = precision_recall_fscore_support(yt, yp, average='binary', zero_division=0)
        f1s.append(f1)
        cm = confusion_matrix(yt, yp)
        if cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()
            sens.append(r)
            specs.append(tn/(tn+fp) if (tn+fp)>0 else np.nan)
        else:
            sens.append(np.nan)
            specs.append(np.nan)
    def ci(arr):
        arr = np.array(arr)
        arr = arr[~np.isnan(arr)]
        if arr.shape[0]==0:
            return (np.nan,np.nan)
        return np.percentile(arr, 2.5), np.percentile(arr, 97.5)
    return ci(accs), ci(aucs), ci(f1s), ci(sens), ci(specs)

feature_file_train = r"C:\Users\zength\Desktop\rad_train_features.xlsx"
feature_file_test1 = r"C:\Users\zength\Desktop\rad_test1_features.xlsx"

label_file_train = r"C:\Users\zength\Desktop\labels_train.csv"
label_file_test1 = r"C:\Users\zength\Desktop\labels_test1.csv"

df_features_train = pd.read_excel(feature_file_train)
df_features_test1 = pd.read_excel(feature_file_test1)

feature_names = df_features_train.columns.tolist()
X_train_all = df_features_train.values.astype('float64')
X_val = df_features_test1.values.astype('float64')

y_train_all = pd.read_csv(label_file_train, encoding="utf-8-sig")["label"].values.astype('int')
y_val = pd.read_csv(label_file_test1, encoding="utf-8-sig")["label"].values.astype('int')

scaler = StandardScaler()
X_train_all = scaler.fit_transform(X_train_all)
X_val = scaler.transform(X_val)




Selected_Lasso_name = [
    'wavelet-HLH_firstorder_Range.2', 'wavelet-HLL_firstorder_Skewness.2',
    'wavelet-HHL_ngtdm_Contrast.2', 'wavelet-HLH_glcm_Imc1', 'log-sigma-5-0-mm-3D_glcm_Imc2'
]
indices = [feature_names.index(name) for name in Selected_Lasso_name]
X_train_all = X_train_all[:, indices]
X_val = X_val[:, indices]

joblib.dump(scaler, '5feature_scaler.joblib')

xgb_param_combinations = list(itertools.product(
    [10,20,25,30], [2,3], [0.01,0.05,0.15], [0.7,1], [0.7,1], [1], [0,1], [1]
))
rf_param_combinations = list(itertools.product([25,50,100], [2,3], [5,7], [2,3]))
lr_param_combinations = list(itertools.product(
    [0.001,0.005,0.015,0.019,0.02,0.021,0.04,0.05,0.07,0.1,0.25,1], ['liblinear'], ['l1','l2']
))
knn_param_combinations = list(itertools.product([3,5,7,9]))
gnb_param_combinations = [(None,)]

best_models = {}
results = {}

def cross_validate_model(model_class, param_grid, X, y, name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
    best_weighted_score = -1
    best_params = None
    best_model = None

    for params in param_grid:
        weighted_scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val_fold = X[train_idx], X[val_idx]
            y_train, y_val_fold = y[train_idx], y[val_idx]

            if name == 'xgb':
                model = model_class(
                    n_estimators=params[0], max_depth=params[1], learning_rate=params[2],
                    subsample=params[3], colsample_bytree=params[4], min_child_weight=params[5],
                    reg_alpha=params[6], reg_lambda=params[7], use_label_encoder=False,
                    eval_metric='logloss', random_state=42, verbose=False
                )
                model.fit(X_train, y_train, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
            elif name == 'rf':
                model = model_class(
                    n_estimators=params[0], max_depth=params[1],
                    min_samples_split=params[2], min_samples_leaf=params[3], random_state=42
                )
                model.fit(X_train, y_train)
            elif name == 'lr':
                model = model_class(
                    C=params[0], solver=params[1], penalty=params[2],
                    max_iter=1000, random_state=42
                )
                model.fit(X_train, y_train)
            elif name == 'knn':
                model = model_class(n_neighbors=params[0])
                model.fit(X_train, y_train)
            elif name == 'gnb':
                model = model_class()
                model.fit(X_train, y_train)
            else:
                continue

            y_pred = model.predict(X_val_fold)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_val_fold)[:, 1]
            else:
                y_proba = y_pred

            try:
                auc = roc_auc_score(y_val_fold, y_proba)
            except:
                auc = 0
            weighted_scores.append(auc)

        avg_weighted_score = np.mean(weighted_scores)
        if avg_weighted_score > best_weighted_score:
            best_weighted_score = avg_weighted_score
            best_params = params
            if name == 'xgb':
                best_model = model_class(
                    n_estimators=params[0], max_depth=params[1], learning_rate=params[2],
                    subsample=params[3], colsample_bytree=params[4], min_child_weight=params[5],
                    reg_alpha=params[6], reg_lambda=params[7], use_label_encoder=False,
                    eval_metric='logloss', random_state=42, verbose=False
                )
            elif name == 'rf':
                best_model = model_class(
                    n_estimators=params[0], max_depth=params[1],
                    min_samples_split=params[2], min_samples_leaf=params[3], random_state=42
                )
            elif name == 'lr':
                best_model = model_class(
                    C=params[0], solver=params[1], penalty=params[2],
                    max_iter=1000, random_state=42
                )
            elif name == 'knn':
                best_model = model_class(n_neighbors=params[0])
            elif name == 'gnb':
                best_model = model_class()

    return best_model, best_params, best_weighted_score

best_models['xgb'], params_xgb, auc_xgb = cross_validate_model(XGBClassifier, xgb_param_combinations, X_train_all, y_train_all, 'xgb')
best_models['rf'], params_rf, auc_rf = cross_validate_model(RandomForestClassifier, rf_param_combinations, X_train_all, y_train_all, 'rf')
best_models['lr'], params_lr, auc_lr = cross_validate_model(LogisticRegression, lr_param_combinations, X_train_all, y_train_all, 'lr')
best_models['knn'], params_knn, auc_knn = cross_validate_model(KNeighborsClassifier, knn_param_combinations, X_train_all, y_train_all, 'knn')
best_models['gnb'], params_gnb, auc_gnb = cross_validate_model(GaussianNB, gnb_param_combinations, X_train_all, y_train_all, 'gnb')


print(f"XGBoost: n_estimators={params_xgb[0]}, max_depth={params_xgb[1]}, learning_rate={params_xgb[2]}")
print(f"Random Forest: n_estimators={params_rf[0]}, max_depth={params_rf[1]}, min_samples_split={params_rf[2]}")
print(f"Logistic Regression: C={params_lr[0]}, solver='{params_lr[1]}', penalty='{params_lr[2]}'")
print(f"KNN: n_neighbors={params_knn[0]}")
print(f"GaussianNB")

for name, model in best_models.items():
    if name == 'xgb':
        model.fit(X_train_all, y_train_all, verbose=False)
    else:
        model.fit(X_train_all, y_train_all)
    joblib.dump(model, f'best_{name}_model.joblib')

for name, model in best_models.items():
    y_pred = model.predict(X_val)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val)[:, 1]
    else:
        y_proba = y_pred
    acc = accuracy_score(y_val, y_pred)
    try:
        auc = roc_auc_score(y_val, y_proba)
    except:
        auc = 0
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_val, y_pred)

    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
        sen = recall
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        sen, spe = 0, 0

    acc_ci, auc_ci, f1_ci, sen_ci, spe_ci = bootstrap_ci(y_val, y_pred, y_proba, n_bootstrap=2000)

    print(f"  Accuracy: {acc:.2f} ({acc_ci[0]:.2f},{acc_ci[1]:.2f})")
    print(f"  AUC: {auc:.2f} ({auc_ci[0]:.2f},{auc_ci[1]:.2f})")
    print(f"  F1: {f1:.2f} ({f1_ci[0]:.2f},{f1_ci[1]:.2f})")
    print(f"  Sensitivity: {sen:.2f} ({sen_ci[0]:.2f},{sen_ci[1]:.2f})")
    print(f"  Specificity: {spe:.2f} ({spe_ci[0]:.2f},{spe_ci[1]:.2f})")
    print(f"  Confusion Matrix:")
    print(cm)
    weighted_score = auc
    results[name] = {'accuracy': acc, 'auc': auc, 'weighted_score': weighted_score, 'f1': f1}

best_model_name = max(results, key=lambda k: results[k]['weighted_score'])

for name, model in best_models.items():
    y_pred_train = model.predict(X_train_all)
    if hasattr(model, "predict_proba"):
        y_proba_train = model.predict_proba(X_train_all)[:, 1]
    else:
        y_proba_train = y_pred_train
    acc_train = accuracy_score(y_train_all, y_pred_train)
    try:
        auc_train = roc_auc_score(y_train_all, y_proba_train)
    except:
        auc_train = 0

    precision_train, recall_train, f1_train, _ = precision_recall_fscore_support(
        y_train_all, y_pred_train, average='binary', zero_division=0)
    cm_train = confusion_matrix(y_train_all, y_pred_train)

    if cm_train.shape == (2,2):
        tn, fp, fn, tp = cm_train.ravel()
        sen_train = recall_train
        spe_train = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        sen_train, spe_train = 0, 0

    acc_ci_tr, auc_ci_tr, f1_ci_tr, sen_ci_tr, spe_ci_tr = bootstrap_ci(
        y_train_all, y_pred_train, y_proba_train, n_bootstrap=2000)

    print(f"  Accuracy: {acc_train:.2f} ({acc_ci_tr[0]:.2f},{acc_ci_tr[1]:.2f})")
    print(f"  AUC: {auc_train:.2f} ({auc_ci_tr[0]:.2f},{auc_ci_tr[1]:.2f})")
    print(f"  F1: {f1_train:.2f} ({f1_ci_tr[0]:.2f},{f1_ci_tr[1]:.2f})")
    print(f"  Sensitivity: {sen_train:.2f} ({sen_ci_tr[0]:.2f},{sen_ci_tr[1]:.2f})")
    print(f"  Specificity: {spe_train:.2f} ({spe_ci_tr[0]:.2f},{spe_ci_tr[1]:.2f})")
    print(f"  Confusion Matrix:")
    print(cm_train)