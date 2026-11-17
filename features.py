# -*- coding: utf-8 -*-
"""
Özellik tanımları ve model girdi dönüşümü.

GERÇEK VERİYE GEÇERKEN: değiştirmen gereken ilk dosya budur.
FEATURES listesi hem arayüzü (form alanları), hem açıklama katmanını
(etiketler, birimler) hem de karşı-olgusal önerileri (hangi özellik değiştirilebilir?) besler.
"""
import numpy as np
import pandas as pd

# Varsayımsal aylık faiz (taksit hesabı için; sadece demo amaçlı)
MONTHLY_RATE = 0.025

# actionable: başvurucunun makul sürede değiştirebileceği bir özellik mi?
#   (yaş, geçmiş gecikmeler vb. değiştirilemez; "öneri" olarak sunulmaz)
# direction: actionable ise hangi yönde değişiklik önerilir ("down" azalt, "up" artır)
# choices: sadece bu değerler mümkünse (ör. vade seçenekleri)
# cf_floor_ratio: karşı-olgusal öneride bu özellik, mevcut değerin en fazla bu oranına kadar düşürülebilir
#   (ör. kredi tutarı: başvurucu ihtiyaç duyduğu tutarın %40'ından azını istemez; "krediyi 10 bin TL'ye indir" anlamsız öneridir)
FEATURES = [
    dict(name="yas", label="Yaş", unit="yıl", kind="int", min=21, max=70, step=1, default=35,
         actionable=False, help="Başvurucunun yaşı"),
    dict(name="aylik_gelir", label="Aylık gelir", unit="TL", kind="int", min=10000, max=200000, step=1000,
         default=45000, actionable=False, help="Net aylık gelir"),
    dict(name="kredi_tutari", label="Talep edilen kredi tutarı", unit="TL", kind="int", min=10000, max=1500000,
         step=5000, default=250000, actionable=True, direction="down", cf_floor_ratio=0.4,
         help="Kredi anaparası"),
    dict(name="vade_ay", label="Vade", unit="ay", kind="int", min=6, max=120, step=6, default=36,
         actionable=True, direction="up", choices=[12, 24, 36, 48, 60, 84, 120], help="Geri ödeme süresi"),
    dict(name="calisma_yili", label="Çalışma süresi", unit="yıl", kind="int", min=0, max=40, step=1, default=6,
         actionable=False, help="Mevcut ve önceki işlerde toplam çalışma süresi"),
    dict(name="kredi_gecmisi_yil", label="Kredi geçmişi uzunluğu", unit="yıl", kind="int", min=0, max=30, step=1,
         default=5, actionable=False, help="İlk kredi/kart kullanımından bu yana geçen süre"),
    dict(name="mevcut_borc", label="Mevcut toplam borç", unit="TL", kind="int", min=0, max=1000000, step=5000,
         default=60000, actionable=True, direction="down", help="Diğer kredi ve kart borçlarının toplamı"),
    dict(name="gecikme_sayisi", label="Son 24 ayda gecikme", unit="adet", kind="int", min=0, max=10, step=1,
         default=0, actionable=False, help="30+ gün gecikmeli ödeme sayısı"),
    dict(name="kart_kullanim_orani", label="Kart limit kullanım oranı", unit="%", kind="int", min=0, max=100,
         step=5, default=35, actionable=True, direction="down", help="Kredi kartı limitinin ne kadarı kullanılıyor"),
    dict(name="teminat_var", label="Teminat", unit="", kind="bool", min=0, max=1, step=1, default=0,
         actionable=True, direction="up", help="Kredi için teminat gösterilebiliyor mu"),
]

FEATURE_NAMES = [f["name"] for f in FEATURES]
FEATURE_BY_NAME = {f["name"]: f for f in FEATURES}

# Model matrisinin sütun adları (Lojistik Regresyon katsayılarını okumak için)
MODEL_COLUMNS = [
    "yas", "log(aylik_gelir)", "log(kredi_tutari)", "vade_ay", "calisma_yili", "kredi_gecmisi_yil",
    "log(1+mevcut_borc)", "gecikme_sayisi", "kart_kullanim_orani", "teminat_var",
    "taksit/gelir orani", "borc/yillik gelir orani",
]


def monthly_installment(amount, months, rate=MONTHLY_RATE):
    """Eşit taksitli kredi aylık ödeme tutarı (annüite)."""
    amount = np.asarray(amount, dtype=float)
    months = np.asarray(months, dtype=float)
    return amount * rate / (1.0 - (1.0 + rate) ** (-months))


def ratios(df):
    """Türetilmiş oranlar: aylık taksidin gelire oranı, borcun yıllık gelire oranı."""
    taksit = monthly_installment(df["kredi_tutari"].to_numpy(float), df["vade_ay"].to_numpy(float))
    taksit_gelir = taksit / df["aylik_gelir"].to_numpy(float)
    borc_gelir = df["mevcut_borc"].to_numpy(float) / (df["aylik_gelir"].to_numpy(float) * 12.0)
    return taksit_gelir, borc_gelir


def model_matrix(X):
    """
    Ham 10 özellik (DataFrame) -> modelin gördüğü 12 sütunlu matris.
    Pipeline içinde FunctionTransformer olarak kullanılır; böylece açıklama katmanı
    sadece 'ham' özelliklerle konuşur, dönüşüm modelin parçası olarak kalır.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=FEATURE_NAMES)
    taksit_gelir, borc_gelir = ratios(X)
    a = X[FEATURE_NAMES].to_numpy(dtype=float)
    out = np.column_stack([
        a[:, 0],                      # yas
        np.log(a[:, 1]),              # log gelir
        np.log(a[:, 2]),              # log kredi tutarı
        a[:, 3],                      # vade
        a[:, 4],                      # çalışma yılı
        a[:, 5],                      # kredi geçmişi
        np.log1p(a[:, 6]),            # log(1+borç)
        a[:, 7],                      # gecikme
        a[:, 8],                      # kart kullanım
        a[:, 9],                      # teminat
        taksit_gelir,
        borc_gelir,
    ])
    return out


def fmt_value(name, v):
    """Arayüz/gerekçe metni için değeri okunur biçimde yaz (Türkçe binlik ayracı)."""
    f = FEATURE_BY_NAME[name]
    v = float(v)
    if f["kind"] == "bool":
        return "Var" if v >= 0.5 else "Yok"
    s = f"{int(round(v)):,}".replace(",", ".")
    if f["unit"] == "TL":
        return s + " TL"
    if f["unit"] == "%":
        return "%" + s
    if f["unit"]:
        return f"{s} {f['unit']}"
    return s
