from sklearn import svm, datasets
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
iris = datasets.load_iris()
x = iris.data[:, 0:2]#using two features only so as to plot decision boundary
y = iris.target
x_train,x_test,y_train,y_test = train_test_split(x, y)
#svc support vector classifier
clf = svm.SVC(kernel = 'linear')#in first case kernel = 'rbf'
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
clf.score(x_test,y_test)
def makegrid(x1, x2, h = 0.02):
    x1_min, x1_max = x1.min() - 1, x1.max() + 1
    x2_min, x2_max = x2.min() - 1, x2.max() + 1
    a = np.arange(x1_min,x1_max,h)
    b = np.arange(x2_min,x2_max,h)
    xx, yy = np.meshgrid(a,b)
    return xx, yy
x1 = np.array([1,3])
x2 = np.array([5,6])
makegrid(x1,x2)
xx, yy = makegrid(x[:,0],x[:,1])
predictions = clf.predict(np.c_[xx.ravel(), yy.ravel()])
plt.scatter(xx.ravel(),yy.ravel(),c = predictions)
plt.show()
xx, yy = makegrid(x[:,0],x[:,1])
predictions = clf.predict(np.c_[xx.ravel(), yy.ravel()])
plt.scatter(xx.ravel(),yy.ravel(),c = predictions)
plt.show()
