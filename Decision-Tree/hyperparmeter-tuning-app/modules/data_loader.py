import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


## Function to create dataset
def load_data():
    """Generates a dummy classifiction dataset with 2 features"""
    iris = load_iris(as_frame=True)
    X=iris.data.iloc[:,2:]
    y=iris.target

    df = pd.DataFrame(data=X, columns=X.columns)
    df['target'] = iris.target

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    print(X_train)
    print(df)
    print(type(X_train))

    return X_train, y_train, X_test, y_test, df

load_data()