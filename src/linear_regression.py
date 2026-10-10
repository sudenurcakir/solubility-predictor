import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression   # Lineer regresyon modeli
from sklearn.metrics import r2_score                 # R² hesaplamak için

df = pd.read_csv("data/features.csv")

X = df[["LogP"]]     # Özellik. Çift köşeli parantez: scikit-learn X'in tablo (2 boyutlu) olmasını ister
y = df["logS"]       # Hedef. Tek parantez: tek boyutlu liste olması yeterli

model = LinearRegression()   # Boş bir model oluştur
model.fit(X, y)              # Modeli eğit: verideki noktalara en iyi uyan çizgiyi bul

egim = model.coef_[0]        # a: coef_ bir liste döner (özellik başına bir eğim), tek özellik var, [0] ile alıyoruz
kesisim = model.intercept_   # b: çizginin y eksenini kestiği nokta

y_tahmin = model.predict(X)  # Her molekül için çizginin verdiği tahmin
r2 = r2_score(y, y_tahmin)   # Gerçek değerlerle tahminleri karşılaştır

print(f"Denklem: logS = {egim:.2f} * LogP + {kesisim:.2f}")   # :.2f = 2 ondalık basamak
print(f"R²: {r2:.3f}")

# ---------- Grafik: Scatter plot + regresyon çizgisi ----------
plt.figure(figsize=(8, 5))
plt.scatter(X, y, alpha=0.4, s=15, label="Moleküller")
# alpha=0.4: noktalar yarı saydam olsun, üst üste binenler koyu görünsün
# s=15: nokta boyutu
plt.plot(X, y_tahmin, color="red", linewidth=2, label="Regresyon çizgisi")   # Tahminleri çizgi olarak çiz
plt.xlabel("LogP")
plt.ylabel("logS (çözünürlük)")
plt.title(f"LogP ve Çözünürlük (R² = {r2:.3f})")   # R²'yi başlığa yaz, grafik tek başına anlamlı olsun
plt.legend()
plt.tight_layout()
plt.savefig("images/04_linear_regression.png", dpi=150)
plt.close()

# ---------- Modelle tahmin yapalım ----------
yeni_logp = pd.DataFrame({"LogP": [0.0, 3.0]})   # Uydurma iki molekül: biri su seven, biri yağ seven
tahminler = model.predict(yeni_logp)
for logp, t in zip(yeni_logp["LogP"], tahminler):   # zip: iki listeyi eşleştirip birlikte dolaşır
    print(f"LogP = {logp} olan molekülün tahmini logS: {t:.2f}")