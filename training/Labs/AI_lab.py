import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib as plt
scaler = StandardScaler()
df = pd.read_csv("F:\PyCharm Community Edition 2022.2.1\Data\iris_labelled(1).csv", header=None)
x = df.loc[:, [2,3]]
y = df.loc[:, 4]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)
print(ppn.coef_)