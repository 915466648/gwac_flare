# coding=utf-8

from sklearn.metrics import classification_report
import joblib


def knn_model(test_dataset):
    print("================KNN Start=====================")
    ## 加载序列数据
    dimen = len(test_dataset[0])
    X_test = test_dataset[:,0:dimen-1]
    Y_test = test_dataset[:, dimen-1]
    print("X_test.shape",X_test.shape)
    print("Y_test.shape",Y_test.shape)

    knn = joblib.load('../cpu_output/s120000r4/KNN0.2606461086637298.pkl')
    y_pred = knn.predict(X_test)
    print(classification_report(Y_test, y_pred, digits=4))
    return y_pred



