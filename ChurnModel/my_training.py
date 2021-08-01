import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

if __name__ == '__main__':
    dataset = pd.read_csv('Churn_Modelling.csv')
    X = dataset.iloc[:, 3:-1].values
    y = dataset.iloc[:, -1].values

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    X[:, 2] = le.fit_transform(X[:, 2])

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    file = open('model.pkl', 'wb')

    # dump information to that file
    pickle.dump(regressor, file)

    file.close()
'''
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    from sklearn.metrics import confusion_matrix, accuracy_score

    y_pred = clf.predict(X_test)
    y_pred = (y_pred > 0.5)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(accuracy_score(y_test, y_pred))

    print(clf.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))

    inf = clf.predict_proba([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])

    print(inf)
    file = open('model.pkl', 'wb')

    # dump information to that file
    pickle.dump(clf, file)

    file.close()
'''