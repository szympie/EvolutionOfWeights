import pandas as pd
from sklearn import model_selection
from comitee import Comitee
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


def read_set(path):
    data = pd.read_csv(path).values
    X = data[:, 1:4]
    Y = data[:, 5]
    return X, Y


X_train, Y_train = read_set('iris/train.csv')
X_test, Y_test = read_set('iris/test.csv')
X_val, Y_val = read_set('iris/val.csv')

classifiers = []
classifiers.append(LogisticRegression())
classifiers.append(LinearDiscriminantAnalysis())
classifiers.append(KNeighborsClassifier())
classifiers.append(DecisionTreeClassifier())
classifiers.append(GaussianNB())
classifiers.append(SVC())

model = Comitee(classifiers, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
kfold = model_selection.KFold(n_splits=10)
model.fit(X_train, Y_train)
print(model.score(X_val, Y_val))
