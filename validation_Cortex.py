import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []
scores = []
for name, model in models:
    score_dict = {}
    model_scores = []
    for fold in range(10):
        train = pd.read_csv(
            "mouses/" + str(fold) + "/train" + str(fold) + ".csv")
        val = pd.read_csv(
            "mouses/" + str(fold) + "/val" + str(fold) + ".csv")

        train = train.values
        X_train = train[:, :77]
        Y_train = train[:, 77]

        val = val.values
        X_val = val[:, :77]
        Y_val = val[:, 77]

        model.fit(X_train, Y_train)
        score = model.score(X_val, Y_val)
        model_scores.append(score)
    score_dict["Model"] = name
    score_dict["Mean Accuracy"] = np.mean(model_scores)
    score_dict["Std Accuracy"] = np.std(model_scores)
    scores.append(score_dict)
scores = pd.DataFrame(scores)
scores.to_csv("cortex_scores_classifiers.csv", index=False,
              columns=["Model", "Mean Accuracy", "Std Accuracy"])
