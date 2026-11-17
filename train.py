# -*- coding: utf-8 -*-
"""
Veriyi üretir, iki modeli eğitir, metrikleri ve global açıklamaları hesaplar, artifacts/ altına kaydeder.

Çalıştırma:  python train.py
Çıktılar:    artifacts/models.joblib, artifacts/meta.json, artifacts/background.csv

İki model bilinçli olarak seçildi:
  lr : Lojistik Regresyon  -> kendi başına yorumlanabilir ("şeffaf") referans model
  gb : Gradient Boosting   -> daha karmaşık, doğrusal olmayan etkileri yakalar; açıklama katmanı gerekir
"""
import json
import os

import joblib
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from data_gen import generate
from features import FEATURE_BY_NAME, FEATURE_NAMES, MODEL_COLUMNS, model_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")

MODEL_TITLES = {"lr": "Lojistik Regresyon", "gb": "Gradient Boosting"}


def build_models():
    lr = Pipeline([
        ("feat", FunctionTransformer(model_matrix)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    gb = Pipeline([
        ("feat", FunctionTransformer(model_matrix)),
        ("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=300,
                                               l2_regularization=1.0, early_stopping=True,
                                               validation_fraction=0.15, n_iter_no_change=25,
                                               random_state=0)),
    ])
    return {"lr": lr, "gb": gb}


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def main(seed=42):
    os.makedirs(ART, exist_ok=True)
    df = generate(n=6000, seed=seed)
    X = df[FEATURE_NAMES]
    y = df["temerrut"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)

    models = build_models()
    meta = dict(
        n_total=int(len(df)), n_train=int(len(X_tr)), n_test=int(len(X_te)),
        default_rate=float(y.mean()), titles=MODEL_TITLES, data_note="Sentetik veri (gerçek değil)",
        metrics={}, roc={}, calibration={}, importance={}, lr_coefficients=[],
    )
    grid = np.linspace(0, 1, 61)

    for key, model in models.items():
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, p)
        meta["metrics"][key] = dict(
            roc_auc=float(roc_auc_score(y_te, p)),
            pr_auc=float(average_precision_score(y_te, p)),
            ks=ks_stat(y_te, p),
            brier=float(brier_score_loss(y_te, p)),
        )
        meta["roc"][key] = [[float(a), float(b)] for a, b in zip(grid, np.interp(grid, fpr, tpr))]
        frac_pos, mean_pred = calibration_curve(y_te, p, n_bins=8, strategy="quantile")
        meta["calibration"][key] = [[float(a), float(b)] for a, b in zip(mean_pred, frac_pos)]

        # Global açıklama: permütasyon önemi (test kümesinde AUC düşüşü). Model-bağımsızdır.
        pi = permutation_importance(model, X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=0)
        imp = [dict(name=n, label=FEATURE_BY_NAME[n]["label"], mean=float(m), std=float(s))
               for n, m, s in zip(FEATURE_NAMES, pi.importances_mean, pi.importances_std)]
        imp.sort(key=lambda r: -r["mean"])
        meta["importance"][key] = imp

    # Lojistik Regresyon: standartlaştırılmış katsayılar (doğrudan yorumlanabilir)
    coefs = models["lr"].named_steps["clf"].coef_[0]
    meta["lr_coefficients"] = sorted(
        [dict(feature=c, coef=float(v)) for c, v in zip(MODEL_COLUMNS, coefs)], key=lambda r: -abs(r["coef"]))

    # Açıklama için referans (arka plan) örneği
    bg = X_tr.sample(300, random_state=0)
    bg.to_csv(os.path.join(ART, "background.csv"), index=False)

    joblib.dump(models, os.path.join(ART, "models.joblib"))
    with open(os.path.join(ART, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    print("Eğitim tamamlandı.")
    for k, m in meta["metrics"].items():
        print(f"  {MODEL_TITLES[k]:20s} AUC={m['roc_auc']:.3f}  PR-AUC={m['pr_auc']:.3f}  "
              f"KS={m['ks']:.3f}  Brier={m['brier']:.3f}")
    return meta


if __name__ == "__main__":
    main()
