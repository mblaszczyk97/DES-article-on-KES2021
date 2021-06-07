import numpy as np
import matplotlib
import pandas as pd
import imblearn
from sklearn.datasets import make_classification
from collections import Counter
from matplotlib import pyplot
from numpy import where

import matplotlib.pyplot as plt
from sklearn.svm import SVC

def train_SVM(df):
   # select the feature columns
   X = df.loc[:, df.columns != 'target']
   # select the label column
   y = df.target

   # train an SVM with linear kernel
   clf = SVC(kernel='linear')
   clf.fit(X, y)

   return clf


def plot_svm_boundary(clf, df, title):
   fig, ax = plt.subplots()
   X0, X1 = df.iloc[:, 0], df.iloc[:, 1]

   x_min, x_max = X0.min() - 1, X0.max() + 1
   y_min, y_max = X1.min() - 1, X1.max() + 1
   xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

   Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
   Z = Z.reshape(xx.shape)
   out = ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

   ax.scatter(X0, X1, c=df.label, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
   ax.set_ylabel('y')
   ax.set_xlabel('x')
   ax.set_title(title)
   plt.show()


df_train = pd.read_csv('../input/coronaVirusCases.csv')

target_count = df_train.target.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

from imblearn.over_sampling import SMOTE
seed = 100
k = 1

df = pd.read_csv('../input/coronaVirusCases.csv')
X = df.loc[:, df.columns != 'target']
y = df.target
sm = SMOTE(sampling_strategy='auto', k_neighbors=k, random_state=seed)
X_res, y_res = sm.fit_resample(X, y)

plt.title('Dataset balanced with synthetic or SMOTE')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(X_res[:, 0], X_res[:, 1], marker='o', c=y_res,
           s=25, edgecolor='k', cmap=plt.cm.coolwarm)
plt.show()

df = pd.concat([pd.DataFrame(X_res), pd.DataFrame(y_res)], axis=1)
# rename the columns
df.columns = ['feature_1', 'feature_2', 'label']
df.to_csv('df_smoted.csv', index=False, encoding='utf-8')

#clf = train_SVM(df_train)
#plot_svm_boundary(clf, df, 'Decision Boundary of SVM trained with a balanced dataset')