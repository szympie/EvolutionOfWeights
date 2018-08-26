import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from comitee import Comitee

classifiers = []
classifiers.append(LogisticRegression())
classifiers.append(LinearDiscriminantAnalysis())
classifiers.append(KNeighborsClassifier())
classifiers.append(DecisionTreeClassifier())
classifiers.append(GaussianNB())
classifiers.append(SVC())

comitee = Comitee(classifiers,
                  [1.0 for _ in range(len(classifiers))],
                  [1.0 for _ in range(8)])
scores = []


def load_set(train_path):
    train = pd.read_csv(train_path)
    train = train.values
    X = train[:, :77]
    Y = train[:, 77]
    return X, Y


for fold in range(10):
    train_path = "mouses/" + str(fold) + "/train" + str(fold) + ".csv"
    test_path = "mouses/" + str(fold) + "/test" + str(fold) + ".csv"
    val_path = "mouses/" + str(fold) + "/val" + str(fold) + ".csv"

    X_train, Y_train = load_set(train_path)
    X_test, Y_test = load_set(test_path)
    X_val, Y_val = load_set(val_path)

    comitee.fit(X_train, Y_train)
    comitee.optimize_classifier_weights(X_test, Y_test, 30, 10)
    scores.append(comitee.score(X_val, Y_val))

print(f"Mean score: {np.mean(scores)}, "
      f"Std score: {np.std(scores)}")
