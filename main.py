# This is a sample Python script.

# Press ⇧F10 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

iris = datasets.load_iris()
parameters = {
    'kernel': ('linear', 'rbf'),
    'C': [1, 10]
}

svc = svm.SVC()
clf = GridSearchCV(svc, parameters)

clf.fit(iris.data, iris.target)
print('like')