import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures   # LogP'den LogP², LogP³ üreten araç
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("data/features.csv")
X = df[["LogP"]]
y = df["logS"]

# Eğriyi düzgün çizmek için LogP'nin en küçüğünden en büyüğüne 200 eşit aralıklı nokta üret.
# Neden? Verideki noktalar karışık sırada; onları doğrudan çizgiyle birleştirirsek
# çizgi ileri geri zıplayarak karmakarışık görünür.
x_cizgi = pd.DataFrame({
    "LogP": np.linspace(X["LogP"].min(), X["LogP"].max(), 200)
})

plt.figure(figsize=(8, 5))
plt.scatter(X, y, alpha=0.3, s=15, color="gray", label="Moleküller")   # Arka planda gerçek veriler

renkler = {1: "red", 2: "green", 3: "blue"}   # Her derece için bir renk (sözlük)

for derece in [1, 2, 3]:   # Üç dereceyi sırayla dene ve karşılaştır
    poly = PolynomialFeatures(degree=derece, include_bias=False)
    # include_bias=False: sabit 1 sütunu ekleme, çünkü LinearRegression kesişimi zaten kendisi ekliyor

    X_poly = poly.fit_transform(X)   # LogP'den yeni sütunlar üret: [LogP, LogP², ...]

    model = LinearRegression()
    model.fit(X_poly, y)             # Yeni sütunlarla normal lineer regresyon

    r2 = r2_score(y, model.predict(X_poly))
    print(f"Derece {derece}: R² = {r2:.3f}")

    # Çizim için: 200 noktayı da aynı dönüşümden geçirip tahmin et
    y_cizgi = model.predict(poly.transform(x_cizgi))
    # Dikkat: burada fit_transform değil transform kullanıyoruz.
    # fit_transform = "öğren ve dönüştür", transform = "zaten öğrendiğin kurala göre dönüştür"

    plt.plot(x_cizgi["LogP"], y_cizgi, color=renkler[derece], linewidth=2,
             label=f"Derece {derece} (R² = {r2:.3f})")

plt.xlabel("LogP")
plt.ylabel("logS (çözünürlük)")
plt.title("Polinom Regresyon: Derece Karşılaştırması")
plt.legend()
plt.tight_layout()
plt.savefig("images/05_polynomial_regression.png", dpi=150)
plt.close()

print("\nGrafik images/05_polynomial_regression.png olarak kaydedildi.")