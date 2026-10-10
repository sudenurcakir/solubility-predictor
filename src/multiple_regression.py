import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler   # Ölçekleme aracı
from sklearn.metrics import r2_score

df = pd.read_csv("data/features.csv")

# Kullanacağımız 7 özelliğin adlarını bir listede topluyoruz.
# Böylece aynı listeyi hem X'i seçerken hem de sonuçları yazdırırken kullanabiliriz.
ozellik_sutunlari = ["MolWt", "LogP", "HDonors", "HAcceptors",
                     "RingCount", "RotatableBonds", "TPSA"]

X = df[ozellik_sutunlari]   # Artık 7 sütunlu bir tablo
y = df["logS"]

# ========== 1. Ölçeklemesiz çoklu regresyon ==========
model = LinearRegression()
model.fit(X, y)
r2 = r2_score(y, model.predict(X))
print(f"Çoklu regresyon R²: {r2:.3f}")

print("\nÖlçeklemesiz katsayılar:")
for ad, katsayi in zip(ozellik_sutunlari, model.coef_):   # Her özellik adını kendi katsayısıyla eşleştir
    print(f"  {ad:15s}: {katsayi:8.4f}")
    # :15s → adı 15 karakterlik alana yaz (sütunlar hizalı görünsün)
    # :8.4f → sayıyı 8 karakterlik alana, 4 ondalıkla yaz

# ========== 2. Ölçekleme ==========
scaler = StandardScaler()
X_olcekli = scaler.fit_transform(X)   # Her sütun için ortalama ve std'yi öğren, sonra dönüştür

print("\nİlk molekül, ölçeklemeden ÖNCE:")
print(X.iloc[0].round(2).to_dict())   # iloc[0]: ilk satır. to_dict(): tek satırda okunaklı yazdır
print("İlk molekül, ölçeklemeden SONRA:")
print(dict(zip(ozellik_sutunlari, X_olcekli[0].round(2))))
# fit_transform sütun adlarını kaybeder, düz bir NumPy dizisi döner. Adları zip ile geri ekliyoruz

# ========== 3. Ölçekli verilerle çoklu regresyon ==========
model_olcekli = LinearRegression()
model_olcekli.fit(X_olcekli, y)
r2_olcekli = r2_score(y, model_olcekli.predict(X_olcekli))
print(f"\nÖlçekli çoklu regresyon R²: {r2_olcekli:.3f}")

# Katsayıları özellik adlarıyla birlikte bir Series'e koyup küçükten büyüğe sırala
katsayilar = pd.Series(model_olcekli.coef_, index=ozellik_sutunlari).sort_values()
print("\nÖlçekli katsayılar (karşılaştırılabilir):")
print(katsayilar.round(3))

# ========== 4. Grafik: Hangi özellik ne kadar etkili? ==========
plt.figure(figsize=(8, 5))
renkler = ["tab:blue" if k < 0 else "tab:red" for k in katsayilar]
# List comprehension: her katsayı için negatifse mavi, pozitifse kırmızı renk seç
plt.barh(katsayilar.index, katsayilar.values, color=renkler)   # barh: yatay bar grafiği
plt.axvline(0, color="black", linewidth=0.8)                     # 0 noktasına dikey çizgi
plt.xlabel("Ölçekli katsayı (etki büyüklüğü)")
plt.title(f"Özelliklerin Çözünürlüğe Etkisi (R² = {r2_olcekli:.3f})")
plt.tight_layout()
plt.savefig("images/06_katsayilar.png", dpi=150)
plt.close()

print("\nGrafik images/06_katsayilar.png olarak kaydedildi.")