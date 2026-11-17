# -*- coding: utf-8 -*-
"""
Sentetik kredi başvurusu verisi üretici.

ÖNEMLİ: Bu veri GERÇEK DEĞİLDİR. Gerçek bir veri seti bulunana kadar hattı uçtan uca
çalıştırmak için üretilmiştir. Gerçek projede bu dosya yerine bir veri yükleme fonksiyonu
(pd.read_csv + temizleme) kullanılacaktır.

Temerrüt olasılığı bilinçli olarak şu mantıkla üretilir (alan bilgisine uygun):
  - taksit/gelir ve borç/gelir oranı yükseldikçe risk artar
  - geçmişte gecikme ve yüksek kart kullanımı riski artırır
  - çalışma süresi, kredi geçmişi, yaş ve teminat riski azaltır
  - doğrusal olmayan iki etki vardır: (yüksek kart kullanımı + gecikme) ve (düşük gelir + yüksek taksit).
    Gradient Boosting bunları yakalayabilirken Lojistik Regresyon yakalayamaz.
  - gözlenmeyen faktörleri temsil eden gürültü eklenir (gerçek hayatta AUC 1.0 olmaz).
"""
import numpy as np
import pandas as pd

from features import FEATURE_NAMES, monthly_installment


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def generate(n=6000, seed=42, target_rate=0.25):
    rng = np.random.default_rng(seed)

    yas = np.clip(rng.normal(38, 10, n), 21, 70).round()
    gelir = np.clip(np.exp(rng.normal(np.log(42000), 0.5, n)), 10000, 200000)
    gelir = (gelir / 500).round() * 500
    calisma = np.clip((yas - 21) * rng.uniform(0.1, 0.9, n), 0, 40).round()
    gecmis = np.clip((yas - 20) * rng.uniform(0.05, 0.7, n), 0, 30).round()
    vade = rng.choice([12, 24, 36, 48, 60, 84, 120], n, p=[.12, .20, .25, .15, .15, .08, .05])
    kredi = np.clip(gelir * rng.uniform(1.5, 14, n), 10000, 1500000)
    kredi = (kredi / 5000).round() * 5000
    borc = np.clip(gelir * 12 * rng.beta(1.5, 4, n) * 1.2, 0, 1000000)
    borc = (borc / 5000).round() * 5000
    lam = 0.4 + 0.9 * (gelir < 25000)
    gecikme = np.minimum(rng.poisson(lam), 10)
    kart = np.clip(rng.beta(2, 3, n) * 100 + gecikme * 3, 0, 100)
    kart = (kart / 5).round() * 5
    teminat = (rng.random(n) < np.clip(0.15 + 0.25 * (kredi > 250000), 0, 1)).astype(int)

    taksit_gelir = monthly_installment(kredi, vade) / gelir
    borc_gelir = borc / (gelir * 12)

    z = (4.0 * (taksit_gelir - 0.25)
         + 1.6 * (borc_gelir - 0.30)
         + 0.55 * gecikme
         + 0.025 * (kart - 40)
         - 0.045 * np.minimum(calisma, 10)
         - 0.05 * np.minimum(gecmis, 15)
         - 0.018 * (yas - 38)
         - 0.9 * teminat
         + 0.9 * ((kart > 80) & (gecikme >= 1))
         + 0.7 * ((gelir < 20000) & (taksit_gelir > 0.35))
         + rng.normal(0, 0.5, n))

    # Ortalama temerrüt oranı target_rate olacak şekilde kesim noktasını bul (ikili arama)
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(z + mid).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    p = _sigmoid(z + (lo + hi) / 2)
    y = (rng.random(n) < p).astype(int)

    df = pd.DataFrame({
        "yas": yas, "aylik_gelir": gelir, "kredi_tutari": kredi, "vade_ay": vade,
        "calisma_yili": calisma, "kredi_gecmisi_yil": gecmis, "mevcut_borc": borc,
        "gecikme_sayisi": gecikme, "kart_kullanim_orani": kart, "teminat_var": teminat,
    })[FEATURE_NAMES].astype(float)
    df["temerrut"] = y
    return df


if __name__ == "__main__":
    d = generate()
    print(d.describe().T[["mean", "min", "max"]].round(2))
    print("Temerrüt oranı:", d["temerrut"].mean().round(3))
