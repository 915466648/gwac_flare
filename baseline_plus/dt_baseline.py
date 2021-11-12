# coding=utf-8
from sklearn.metrics import classification_report
import joblib

def dt_model(test_dataset):
    print("================Decision Tree Start=====================")

    X_test = test_dataset[:, 0:13]
    Y_test = test_dataset[:, 13]
    print("X_test.shape",X_test.shape)
    print("Y_test.shape",Y_test.shape)
    reg = joblib.load('../cpu_output/s120000r4/DecisionTreeClassifier0.995464991573464.pkl')
    Y_pred = reg.predict(X_test)
    print(classification_report(Y_test, Y_pred, digits=4))
    return Y_pred