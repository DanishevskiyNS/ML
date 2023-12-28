import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tree import CART
from forest import RandomForest

def get_accuracy(y_true, y_preds):
    correct = np.sum(y_preds == y_true)

    return correct / y_true.shape[0]


if __name__ == '__main__':

    df = pd.read_csv("data/cars/car.data", header=None).values

    df_train, df_test = train_test_split(df, test_size=0.3, shuffle=True)

    tree_1 = CART(max_depth=100)
    tree_1.fit(df_train)

    y_preds_1 = tree_1.predict(df_test[:, :-1])

    print(get_accuracy(df_test[:, -1], y_preds_1))

    forest = RandomForest(30, 200, 100)
    forest.fit(df_train)
    y_preds = forest.predict(df_test[:, :-1])
    
    
    print(get_accuracy(df_test[:, -1], y_preds))

    print(classification_report(df_test[:, -1], y_preds))

