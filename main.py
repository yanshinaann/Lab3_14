import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier


# const definition


WELFARE = {1: "", 2: ""}

filename = "/Users/anna/Documents/Documents/University /4 курс/ИнтСИС/Lab3_14/wisc_bc_data.data"
df = pd.read_table(filename, sep=',')

fd = df
test = 0.3
X = fd.values[:, 2:]
y = fd.values[:, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test)

acc = []

for k in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    res = pd.DataFrame()
    res['test'] = y_test
    res['pred'] = y_pred
    data_crosstab = pd.crosstab(res['test'],
                                res['pred'],
                                margins=False)

    acc.append(metrics.accuracy_score(y_test, y_pred))
    print(data_crosstab)

plt.plot(range(1, 30), acc)
plt.show()
