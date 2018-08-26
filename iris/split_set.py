import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

iris = pd.read_csv('Iris_label_encoded.csv')

y = iris['Species']
X = iris.drop('Species', axis=1)

kf = StratifiedShuffleSplit(n_splits=10, train_size=0.1, test_size=0.9)
for index, split in enumerate(kf.split(X, y)):
    train = iris.iloc[split[0]]
    test_all = iris.iloc[split[1]]
    test = test_all.sample(frac=0.5).reset_index(drop=True)
    val = test_all.sample(frac=0.5).reset_index(drop=True)

    train.to_csv("./" + str(index) + "/train" + str(index) + ".csv",
                 index=False)
    test.to_csv("./" + str(index) + "/test" + str(index) + ".csv", index=False)
    val.to_csv("./" + str(index) + "/val" + str(index) + ".csv", index=False)
