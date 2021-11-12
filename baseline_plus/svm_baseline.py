# coding=utf-8

from sklearn.metrics import classification_report
import joblib
from sklearn.metrics import fbeta_score

def svm_model(test_dataset):
    print("================SVM Start=====================")
    X_test = test_dataset[:, 0:13]
    Y_test = test_dataset[:, 13]
    print("X_test.shape",X_test.shape)
    print("Y_test.shape",Y_test.shape)

    clf = joblib.load('../cpu_output/s120000r4/SVM-rbf0.666058394160584.pkl')
    Y_pred = clf.predict(X_test)
    f2 = fbeta_score(Y_test, Y_pred, beta=2)
    print("f2",f2)
    print("classification_report on sub sequences")
    print(classification_report(Y_test,Y_pred, digits=4))
    return Y_pred