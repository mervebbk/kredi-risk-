# Açıklanabilir Kredi Riski Tahmini ve Karar Destek Sistemi (Demo)

Bitirme projesi için hazırlanmış çalışan bir demo iskeleti. Bir kredi başvurusu için:

1. **Risk tahmini** yapar (temerrüt olasılığı, iki farklı model),
2. **Nedenini açıklar** (Shapley değerleri: hangi özellik riski kaç puan artırdı/azalttı),
3. **"Ne değişirse karar değişir?"** sorusunu cevaplar (karşı-olgusal öneriler),
4. **Karar destek** sunar (risk bandı, ayarlanabilir eşikler, beklenen kayıp, gerekçe notu).

> ⚠️ **Veri sentetiktir (gerçek değildir).** Bu sürüm, uçtan uca hattı göstermek için yapay veriyle çalışır.
> Gerçek bir veri setine geçiş adımları aşağıdadır. Metrikler gerçek dünyayı temsil etmez.

## Hızlı başlangıç

Python 3.10+ gerekir.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

python train.py        # (isteğe bağlı) modelleri eğitir; app ilk açılışta kendisi de yapar
python app.py          # http://127.0.0.1:5000 adresini tarayıcıda aç
```

Testler: `python -m unittest discover -s tests -v`

İpucu (sunum için): `http://127.0.0.1:5000/?preset=yuksek&model=lr` ve `http://127.0.0.1:5000/#model` gibi
adresler doğrudan ilgili örnek/model/sekmeyi açar.

## Dosya yapısı

| Dosya | Görev |
|---|---|
| `features.py` | **Özellik tanımları** (ad, etiket, aralık, "değiştirilebilir mi?"), model girdi dönüşümü |
| `data_gen.py` | Sentetik veri üreticisi (gerçek veriyle değiştirilecek) |
| `train.py` | Modelleri eğitir; metrik, ROC, kalibrasyon, permütasyon önemi hesaplar; `artifacts/` yazar |
| `explain.py` | **Açıklama ve karar katmanı**: Shapley örnekleme, karşı-olgusal arama, karar bantları, beklenen kayıp |
| `app.py` | Flask sunucusu ve API |
| `templates/index.html` | Arayüz (dış kütüphane yok, internetsiz çalışır) |
| `tests/test_explain.py` | 15 otomatik test |

## Gerçek veri setine geçiş (önerilen sıra)

1. Veriyi `data/` altına koy; `data_gen.py` yerine `pd.read_csv` ile okuyan bir fonksiyon yaz (hedef sütun: `temerrut`, 0/1).
2. `features.py` içindeki `FEATURES` listesini yeni sütunlara göre güncelle. Her özellik için:
   `actionable` (başvurucu değiştirebilir mi?) ve gerekirse `direction` / `choices` / `cf_floor_ratio` belirle.
3. `model_matrix` fonksiyonunu yeni özelliklere göre düzenle (kategorikler için one-hot, oranlar vb.).
4. `python train.py` → metrikleri kontrol et. **AUC 0.95+ çıkıyorsa veri sızıntısından şüphelen.**
5. `python -m unittest discover -s tests -v` → açıklama testleri hâlâ geçmeli (özellikle `sum(phi) = tahmin - taban`).
6. İsteğe bağlı: `pip install shap` ve `explain.py`'daki Shapley sonuçlarını `shap.Explainer` ile karşılaştır.

## Yöntem özeti

* **Modeller:** Lojistik Regresyon (şeffaf) ve HistGradientBoosting (karmaşık). İkisi de aynı `Pipeline` yapısında.
* **Yerel açıklama:** Permütasyon örneklemeli Shapley değerleri (300 örnek, sabit tohum). Katkıların toplamı `tahmin − referans risk`
  farkına tam eşittir.
* **Global açıklama:** Test kümesinde permütasyon önemi (AUC düşüşü) + Lojistik Regresyon katsayıları.
* **Karşı-olgusal:** Yalnızca değiştirilebilir özellikler; önce tek, sonra ikili, sonra çoklu değişiklik aranır; en küçük değişiklik önerilir.
* **Karar destek:** İki ayarlanabilir eşik (onay / ret), beklenen kayıp = PD × LGD × EAD (LGD varsayımsaldır), otomatik gerekçe metni.

## Sınırlılıklar

* Sentetik veri; gerçek müşteri davranışını temsil etmez.
* Shapley açıklamaları modelin davranışını anlatır, nedensellik iddia etmez.
* Adalet analizi (cinsiyet/yaş gruplarına göre hata oranları) bu sürümde yoktur; yol haritasının 9. adımıdır.
* Faiz oranı (aylık %2,5) ve LGD değerleri varsayımdır.
* Bu bir eğitim prototipidir; gerçek kredi kararlarında kullanılamaz.
