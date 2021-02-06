
# 1.İrisi veri setinin yüklenmesi
from sklearn.datasets import load_iris
iris_dataset = load_iris()
print(iris_dataset)



# 2. Verinin ikiye bölünmesi
from sklearn.model_selection import train_test_split
# x_ogren, x_test, y_ogren, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size = 0,33)  burada son belirttiğimiz kısımda test datasını %33 dilime ayırması içindi. 
X_ogren, X_test, y_ogren, y_test = train_test_split(iris_dataset['data'], iris_dataset['target']) # Bu şekilde yaptığımızda %75 data, %25 test şeklinde otomatik ayırıyor.
print(X_ogren.shape)
print(X_test.shape)



# 3. Uygun modeli seçme
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
print(knn)



# 4. Ogrenme
knn.fit(X_ogren, y_ogren)



# 5. Tahmin
X_yeni = [[3.5, 2.1, 3.4, 1.2]]
tahmin = knn.predict(X_yeni)
print(tahmin)



# 6. Dogruluk & Test verisi
dogruluk = knn.predict(X_test)
print(dogruluk)

import numpy as np
print(np.mean(dogruluk == y_test) * 100)




















