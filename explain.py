# -*- coding: utf-8 -*-
"""
Açıklanabilirlik ve karar destek katmanı (model-bağımsız, sadece predict_proba'ya ihtiyaç duyar).

Üç bileşen:
  1) shapley_sampling  : Bir başvurunun tahminini özelliklere paylaştırır (yerel açıklama).
  2) counterfactuals   : "Hangi değişiklikler kararı değiştirir?" (karşı-olgusal öneriler).
  3) decide / narrative: Risk bandı, beklenen kayıp ve Türkçe gerekçe metni (karar destek).

Not: 'shap' kütüphanesi bu ortamda kurulu olmadığı için Shapley değerleri, Strumbelj & Kononenko (2014)
tarzı permütasyon örneklemesiyle hesaplanır. SHAP'ın temel fikriyle aynıdır; gerçek projede
'shap' kurulabiliyorsa sonuçlar karşılaştırılmalıdır.
"""
import itertools
import numpy as np
import pandas as pd

from features import FEATURES, FEATURE_BY_NAME, FEATURE_NAMES, fmt_value


# ----------------------------------------------------------------------------------------------
# 1) Yerel açıklama: Shapley değerleri (permütasyon örneklemesi)
# ----------------------------------------------------------------------------------------------
def shapley_sampling(predict_fn, x, background, n_samples=300, seed=0):
    """
    predict_fn : DataFrame -> olasılık vektörü (temerrüt olasılığı)
    x          : dict, açıklanacak başvuru (ham özellikler)
    background : DataFrame, referans (eğitim verisinden örnek)
    Dönüş      : phi (özellik başına katkı, olasılık birimi), base (referans ortalama risk), fx (tahmin)

    Fikir: Rastgele bir referans başvuru (background satırı) seç, rastgele bir özellik sırası seç.
    Özellikleri sırayla referanstan "x'inkine" çevir; her çevirmenin tahmini ne kadar değiştirdiğini
    o özelliğe yaz. Çok sayıda örneğin ortalaması Shapley değeridir.
    Özellik: sum(phi) = fx - base (tam eşitlik; testlerde doğrulanır).
    """
    cols = list(background.columns)
    d = len(cols)
    rng = np.random.default_rng(seed)  # sabit tohum: aynı girdi -> aynı açıklama (kaymaz)
    bg = background.to_numpy(float)
    xv = np.array([float(x[c]) for c in cols])

    idx = rng.integers(0, len(bg), n_samples)
    B = bg[idx]
    order = np.argsort(rng.random((n_samples, d)), axis=1)  # her satır bir rastgele permütasyon
    rows = np.arange(n_samples)

    paths = np.empty((n_samples, d + 1, d))
    cur = B.copy()
    paths[:, 0, :] = cur
    for k in range(d):
        j = order[:, k]
        cur = cur.copy()
        cur[rows, j] = xv[j]
        paths[:, k + 1, :] = cur

    p = predict_fn(pd.DataFrame(paths.reshape(-1, d), columns=cols)).reshape(n_samples, d + 1)
    diffs = p[:, 1:] - p[:, :-1]
    phi = np.zeros(d)
    np.add.at(phi, order.ravel(), diffs.ravel())
    phi /= n_samples
    base = float(p[:, 0].mean())
    fx = float(predict_fn(pd.DataFrame([xv], columns=cols))[0])
    return phi, base, fx


def explanation_rows(x, phi):
    """Arayüz için: özellik, değer, katkı (yüzde puan) – mutlak katkıya göre sıralı."""
    rows = []
    for name, ph in zip(FEATURE_NAMES, phi):
        f = FEATURE_BY_NAME[name]
        rows.append(dict(name=name, label=f["label"], value=float(x[name]),
                         display=fmt_value(name, x[name]), phi_pp=float(ph * 100.0)))
    rows.sort(key=lambda r: -abs(r["phi_pp"]))
    return rows


# ----------------------------------------------------------------------------------------------
# 2) Karşı-olgusal öneriler
# ----------------------------------------------------------------------------------------------
def _candidate_values(f, current, n_steps=40):
    """Bir özellik için, mevcut değerden başlayıp 'önerilen yönde' denenecek aday değerler."""
    lo, hi = f["min"], f["max"]
    if f["direction"] == "down" and f.get("cf_floor_ratio"):
        lo = max(lo, current * f["cf_floor_ratio"])  # gerçekçi alt sınır
    if f.get("choices"):
        vals = [v for v in f["choices"] if (v > current if f["direction"] == "up" else v < current)]
        return sorted(vals) if f["direction"] == "up" else sorted(vals, reverse=True)
    if f["kind"] == "bool":
        return [1.0] if (f["direction"] == "up" and current < 0.5) else []
    if f["direction"] == "down":
        if current <= lo:
            return []
        vals = np.linspace(current, lo, n_steps + 1)[1:]
    else:
        if current >= hi:
            return []
        vals = np.linspace(current, hi, n_steps + 1)[1:]
    step = f["step"]
    vals = np.unique(np.round(vals / step) * step)
    vals = [float(v) for v in vals if v != current]
    return sorted(vals, reverse=(f["direction"] == "down"))


def _effort(f, old, new):
    """Değişikliğin 'zorluğu' = aralığa göre bağıl değişim (0-1)."""
    return abs(new - old) / float(f["max"] - f["min"])


def counterfactuals(predict_fn, x, target, max_results=3):
    """
    Risk, 'target' altına inecek en küçük değişiklikleri arar. Sadece FEATURES içinde actionable=True
    olan özellikler değiştirilir (yaş, geçmiş gecikme gibi değiştirilemeyenler asla önerilmez).

    Dönüş: dict(
      hedef, mevcut_risk,
      oneriler=[{degisiklikler:[{name,label,from,to,from_text,to_text}], yeni_risk, caba}],
      ulasilabilir (bool), en_iyi_risk
    )
    """
    base_row = {k: float(x[k]) for k in FEATURE_NAMES}
    p0 = float(predict_fn(pd.DataFrame([base_row]))[0])
    act = [f for f in FEATURES if f["actionable"]]

    def change(f, new):
        return dict(name=f["name"], label=f["label"], **{"from": base_row[f["name"]], "to": float(new)},
                    from_text=fmt_value(f["name"], base_row[f["name"]]), to_text=fmt_value(f["name"], new))

    if p0 < target:  # zaten hedefte: öneri gerekmez
        return dict(hedef=float(target), mevcut_risk=p0, oneriler=[], ulasilabilir=False,
                    zaten_hedefte=True, en_iyi_risk=p0)

    results = []

    # (a) Tek özellik değişiklikleri: hedefe ulaşan EN KÜÇÜK değişiklik
    for f in act:
        cands = _candidate_values(f, base_row[f["name"]])
        if not cands:
            continue
        df = pd.DataFrame([base_row] * len(cands))
        df[f["name"]] = cands
        probs = predict_fn(df)
        for v, pr in zip(cands, probs):  # adaylar küçük değişimden büyüğe sıralı
            if pr < target:
                results.append(dict(degisiklikler=[change(f, v)], yeni_risk=float(pr),
                                    caba=_effort(f, base_row[f["name"]], v)))
                break

    # (b) Tek özellik yetmiyorsa ikili kombinasyonlar (kaba ızgara)
    if not results:
        for fa, fb in itertools.combinations(act, 2):
            ca = _candidate_values(fa, base_row[fa["name"]])
            cb = _candidate_values(fb, base_row[fb["name"]])
            if not ca or not cb:
                continue
            ca = ca[:: max(1, len(ca) // 8)] + [ca[-1]]
            cb = cb[:: max(1, len(cb) // 8)] + [cb[-1]]
            combos = list(itertools.product(ca, cb))
            df = pd.DataFrame([base_row] * len(combos))
            df[fa["name"]] = [c[0] for c in combos]
            df[fb["name"]] = [c[1] for c in combos]
            probs = predict_fn(df)
            best = None
            for (va, vb), pr in zip(combos, probs):
                if pr < target:
                    e = _effort(fa, base_row[fa["name"]], va) + _effort(fb, base_row[fb["name"]], vb)
                    if best is None or e < best[0]:
                        best = (e, va, vb, float(pr))
            if best:
                results.append(dict(degisiklikler=[change(fa, best[1]), change(fb, best[2])],
                                    yeni_risk=best[3], caba=best[0]))

    # (c) İkili de yetmezse: açgözlü çoklu değişiklik (en çok düşüren özelliği uç değere çek),
    #     hedefe inince her değişikliği "hâlâ hedefte kalacak en küçük değere" geri gevşet.
    if not results:
        cur = dict(base_row)
        chosen, remaining = [], list(act)
        while remaining and len(chosen) < 4 and float(predict_fn(pd.DataFrame([cur]))[0]) >= target:
            best = None
            for f in remaining:
                cands = _candidate_values(f, base_row[f["name"]])
                if not cands:
                    continue
                row = dict(cur)
                row[f["name"]] = cands[-1]
                pr = float(predict_fn(pd.DataFrame([row]))[0])
                if best is None or pr < best[0]:
                    best = (pr, f, cands[-1])
            if best is None:
                break
            cur[best[1]["name"]] = best[2]
            chosen.append(best[1])
            remaining.remove(best[1])
        if float(predict_fn(pd.DataFrame([cur]))[0]) < target:
            for f in reversed(chosen):
                cands = _candidate_values(f, base_row[f["name"]])
                df = pd.DataFrame([cur] * len(cands))
                df[f["name"]] = cands
                for v, pr in zip(cands, predict_fn(df)):
                    if pr < target:
                        cur[f["name"]] = v
                        break
            results.append(dict(
                degisiklikler=[change(f, cur[f["name"]]) for f in chosen],
                yeni_risk=float(predict_fn(pd.DataFrame([cur]))[0]),
                caba=sum(_effort(f, base_row[f["name"]], cur[f["name"]]) for f in chosen)))

    results.sort(key=lambda r: r["caba"])
    results = results[:max_results]

    # En iyi durum: tüm eyleme dönüştürülebilir özellikler "en iyi" uçta ise risk ne olur?
    best_row = dict(base_row)
    for f in act:
        cands = _candidate_values(f, base_row[f["name"]])
        if cands:
            best_row[f["name"]] = cands[-1]
    best_risk = float(predict_fn(pd.DataFrame([best_row]))[0])

    return dict(hedef=float(target), mevcut_risk=p0, oneriler=results, ulasilabilir=bool(results),
                zaten_hedefte=False, en_iyi_risk=min(best_risk, p0))


# ----------------------------------------------------------------------------------------------
# 3) Karar destek: risk bandı, beklenen kayıp, gerekçe metni
# ----------------------------------------------------------------------------------------------
LGD_TEMINATSIZ = 0.45   # Temerrüt halinde kayıp oranı (varsayım; Basel'de bireysel kredi için tipik değerler)
LGD_TEMINATLI = 0.25


def decide(p, low, high):
    """Politika eşiklerine göre öneri. Nihai karar her zaman analistindir."""
    if p < low:
        return dict(band="dusuk", band_text="Düşük risk", action="Onay önerilir",
                    detail="Risk, onay eşiğinin altında. Standart doğrulama sonrası onay düşünülebilir.")
    if p < high:
        return dict(band="orta", band_text="Orta risk", action="Manuel inceleme önerilir",
                    detail="Risk, onay ve ret eşikleri arasında. Analist ek belge/teminat talep edebilir.")
    return dict(band="yuksek", band_text="Yüksek risk", action="Ret önerilir (analist onayıyla)",
                detail="Risk, ret eşiğinin üstünde. Karar, analist incelemesi ve gerekçe notuyla kesinleşir.")


def expected_loss(p, x):
    """Beklenen kayıp = PD × LGD × EAD (PD: model olasılığı, EAD: kredi tutarı)."""
    lgd = LGD_TEMINATLI if float(x["teminat_var"]) >= 0.5 else LGD_TEMINATSIZ
    ead = float(x["kredi_tutari"])
    return dict(pd=float(p), lgd=lgd, ead=ead, el=float(p * lgd * ead))


def _pp(v):
    return ("%+.1f" % v).replace(".", ",")


def narrative(p, base, rows, decision):
    """Analist için otomatik Türkçe gerekçe cümlesi."""
    inc = [r for r in rows if r["phi_pp"] > 0.25][:3]
    dec = [r for r in rows if r["phi_pp"] < -0.25][:3]
    pct = lambda v: ("%.1f" % (v * 100)).replace(".", ",")
    s = (f"Modelin tahmini temerrüt riski %{pct(p)}; referans (ortalama başvuru) risk %{pct(base)}. "
         f"{decision['band_text']} bandında: {decision['action'].lower()}. ")
    if inc:
        s += "Riski en çok artıranlar: " + ", ".join(
            f"{r['label'].lower()} ({r['display']}, {_pp(r['phi_pp'])} puan)" for r in inc) + ". "
    if dec:
        s += "Riski en çok azaltanlar: " + ", ".join(
            f"{r['label'].lower()} ({r['display']}, {_pp(r['phi_pp'])} puan)" for r in dec) + "."
    return s.strip()
