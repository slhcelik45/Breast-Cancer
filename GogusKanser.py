import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

veri = pd.read_csv("breast-cancer-wisconsin.data")
veri.replace('?', -99999, inplace='true')# verideki ? olan yerleri güncellenir.
veri.drop(['id'], axis=1) #Verideki ID kısmını atıyoruz.

y = veri.benormal # veridekibenormal değerini y atıyoruz
x = veri.drop(['benormal'], axis=1) # x değişkenine benormal cıkarıp günceliyoruz.
imp = Imputer(missing_values=-99999, strategy="mean",axis=0)# bilinmeyen degerlerin ortalamsını hesaplıyoruz.
x = imp.fit_transform(x)# kayıp yerlerin ortalma degeri atıyoruz.

tahmin = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto',
                    leaf_size=30, p=2, metric='euclidean', metric_params=None, n_jobs=1)
tahmin.fit(x,y)
ytahmin = tahmin.predict(x)

basari = accuracy_score(y, ytahmin, normalize=True, sample_weight=None)
print("Yüzde",basari*100," oranında:" )

print(tahmin.predict([1,2,2,2,3,2,1,2,3,2]))

"""
bu kısımda k parametresinin en optimun değerini hesaplıyorum.
for z in range(25):
    z = 2*z+1
    print("En yakın",z,"komşu kullandığımızda tutarlılık oranımız")
    tahmin = KNeighborsClassifier(n_neighbors=z, weights='uniform', algorithm='auto',
                                  leaf_size=30, p=2, metric='euclidean', metric_params=None, n_jobs=1)
    tahmin.fit(x,y)
    ytahmin = tahmin.predict(x)

    basari = accuracy_score(y, ytahmin, normalize=True, sample_weight=None)
    print(basari)
 """