# -*- coding: utf-8 -*-
"""
Açıklanabilir Kredi Riski Karar Destek Sistemi – Flask demo uygulaması.

Çalıştırma:   python app.py        ->  http://127.0.0.1:5000
İlk çalıştırmada artifacts/ yoksa modeller otomatik eğitilir.

Uç noktalar:
  GET  /               arayüz
  GET  /api/meta       özellik tanımları, metrikler, global önem, ROC, kalibrasyon, örnek başvurular
  POST /api/predict    {features, model, low, high} -> risk, bant, açıklama, karşı-olgusal, beklenen kayıp
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from explain import (counterfactuals, decide, expected_loss, explanation_rows, narrative,
                     shapley_sampling)
from features import FEATURES, FEATURE_BY_NAME, FEATURE_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")

PRESETS = [
    dict(id="dusuk", title="Düşük riskli başvuru",
         values=dict(yas=42, aylik_gelir=85000, kredi_tutari=150000, vade_ay=36, calisma_yili=14,
                     kredi_gecmisi_yil=12, mevcut_borc=20000, gecikme_sayisi=0, kart_kullanim_orani=15,
                     teminat_var=1)),
    dict(id="orta", title="Orta riskli başvuru",
         values=dict(yas=31, aylik_gelir=38000, kredi_tutari=280000, vade_ay=36, calisma_yili=4,
                     kredi_gecmisi_yil=4, mevcut_borc=90000, gecikme_sayisi=1, kart_kullanim_orani=55,
                     teminat_var=0)),
    dict(id="yuksek", title="Yüksek riskli başvuru",
         values=dict(yas=29, aylik_gelir=28000, kredi_tutari=260000, vade_ay=36, calisma_yili=2,
                     kredi_gecmisi_yil=3, mevcut_borc=70000, gecikme_sayisi=2, kart_kullanim_orani=80,
                     teminat_var=0)),
]

app = Flask(__name__)
app.json.ensure_ascii = False


def load_artifacts():
    if not os.path.exists(os.path.join(ART, "models.joblib")):
        import train
        train.main()
    models = joblib.load(os.path.join(ART, "models.joblib"))
    with open(os.path.join(ART, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    background = pd.read_csv(os.path.join(ART, "background.csv"))[FEATURE_NAMES]
    return models, meta, background


MODELS, META, BACKGROUND = load_artifacts()


def clean_features(raw):
    """Gelen değerleri doğrula, aralığa kırp. Eksik/bozuk değer -> ValueError."""
    if not isinstance(raw, dict):
        raise ValueError("features nesne olmalı")
    out = {}
    for f in FEATURES:
        if f["name"] not in raw:
            raise ValueError("Eksik alan: " + f["name"])
        try:
            v = float(raw[f["name"]])
        except (TypeError, ValueError):
            raise ValueError("Sayısal olmayan değer: " + f["name"])
        if not np.isfinite(v):
            raise ValueError("Geçersiz değer: " + f["name"])
        out[f["name"]] = float(min(max(v, f["min"]), f["max"]))
    return out


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/meta")
def api_meta():
    return jsonify(dict(features=FEATURES, presets=PRESETS, **META))


@app.post("/api/predict")
def api_predict():
    body = request.get_json(silent=True) or {}
    try:
        x = clean_features(body.get("features"))
        key = body.get("model", "gb")
        if key not in MODELS:
            raise ValueError("Bilinmeyen model: %s" % key)
        low = float(body.get("low", 0.20))
        high = float(body.get("high", 0.45))
        if not (0.01 <= low < high <= 0.99):
            raise ValueError("Eşikler 0.01 <= düşük < yüksek <= 0.99 olmalı")
    except ValueError as e:
        return jsonify(error=str(e)), 400

    model = MODELS[key]
    predict_fn = lambda df: model.predict_proba(df[FEATURE_NAMES])[:, 1]

    phi, base, p = shapley_sampling(predict_fn, x, BACKGROUND, n_samples=300, seed=0)
    rows = explanation_rows(x, phi)
    decision = decide(p, low, high)
    cf = counterfactuals(predict_fn, x, target=low)

    return jsonify(dict(
        model=key, model_title=META["titles"][key],
        risk=p, base=base, low=low, high=high,
        decision=decision,
        expected_loss=expected_loss(p, x),
        contributions=rows,
        narrative=narrative(p, base, rows, decision),
        counterfactual=cf,
        phi_sum_check=float(phi.sum() - (p - base)),
    ))


@app.get("/api/health")
def health():
    return jsonify(status="ok", models=list(MODELS.keys()))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
