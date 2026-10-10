import pandas as pd                
import numpy as np                 
import matplotlib.pyplot as plt    
import seaborn as sns              


df = pd.read_csv("data/features.csv")   # Adım 1'de oluşturduğumuz tabloyu oku
logS = df["logS"]                       # Hedef sütunu ayrı bir değişkene al

# ---------- Konu 1: Merkezi eğilim ----------
print("=== Merkezi Eğilim ===")
print("Ortalama:", round(logS.mean(), 2))         # round(x, 2): 2 ondalık basamağa yuvarla
print("Medyan:", round(logS.median(), 2))
print("Mod:", logS.round(0).mode()[0])            # Değerler ondalıklı olduğu için tam sayıya yuvarlayıp
                                                  # en sık tekrar edeni buluyoruz. mode() liste döner, [0] ilkini alır

# ---------- Konu 2: Yayılım ----------
print("\n=== Yayılım ===")
print("Standart sapma:", round(logS.std(), 2))
print("Varyans:", round(logS.var(), 2))
print("En küçük:", round(logS.min(), 2))          # En az çözünen molekül
print("En büyük:", round(logS.max(), 2))          # En çok çözünen molekül

# ---------- Konu 3: Yüzdelikler ----------
print("\n=== Yüzdelikler ===")
for p in [25, 50, 75, 90]:                        # Bu 4 yüzdeliği sırayla hesapla
    deger = np.percentile(logS, p)                # Verinin %p'si bu değerin altında
    print(f"{p}. yüzdelik: {round(deger, 2)}")

# ---------- Grafik 1: Histogram (dağılımın şekli) ----------
plt.figure(figsize=(8, 5))                        # Yeni boş grafik, 8x5 inç boyutunda
plt.hist(logS, bins=30, edgecolor="black")        # bins=30: değer aralığını 30 kutuya böl
plt.axvline(logS.mean(), color="red", linestyle="--", label="Ortalama")    # Ortalamaya dikey çizgi
plt.axvline(logS.median(), color="green", linestyle="--", label="Medyan")  # Medyana dikey çizgi
plt.xlabel("logS (çözünürlük)")                   # X ekseni etiketi
plt.ylabel("Molekül sayısı")                      # Y ekseni etiketi
plt.title("Çözünürlük Dağılımı")                  # Grafik başlığı
plt.legend()                                      # Çizgilerin açıklama kutusunu göster
plt.tight_layout()                                # Yazıların kesilmemesi için kenar boşluklarını ayarla
plt.savefig("images/01_logS_histogram.png", dpi=150)   # Kaydet. dpi=150: README için yeterli kalite
plt.close()                                       # Grafiği kapat ki sonraki grafikle karışmasın

# ---------- Grafik 2: Box plot (aykırı değerler) ----------
plt.figure(figsize=(8, 3))
sns.boxplot(x=logS, color="skyblue")              # Kutunun kenarları 25. ve 75. yüzdelik, ortadaki çizgi medyan.
                                                  # Bıyıkların dışındaki noktalar aykırı değer
plt.xlabel("logS (çözünürlük)")
plt.title("Çözünürlük Box Plot")
plt.tight_layout()
plt.savefig("images/02_logS_boxplot.png", dpi=150)
plt.close()

# ---------- Grafik 3: Korelasyon ısı haritası ----------
sayisal = df.drop(columns=["smiles"])             # smiles metin olduğu için korelasyona katılamaz, çıkar
korelasyon = sayisal.corr()                       # Her sütun çifti arasındaki korelasyon (-1 ile +1 arası)
plt.figure(figsize=(8, 6))
sns.heatmap(korelasyon, annot=True, fmt=".2f", cmap="coolwarm", center=0)
# annot=True: kutuların içine sayıyı yaz
# fmt=".2f": 2 ondalık basamak
# cmap="coolwarm": negatif mavi, pozitif kırmızı
# center=0: 0 değeri beyaz olsun
plt.title("Özellikler Arası Korelasyon")
plt.tight_layout()
plt.savefig("images/03_korelasyon.png", dpi=150)
plt.close()

print("\nGrafikler images/ klasörüne kaydedildi.")