# coding=utf-8

from sklearn.metrics import fbeta_score,classification_report
from sklearn import svm

SEED = 6666666

batch_size, time_steps, input_dim = None, 188, 1

def tcn_svm_model(new_train,new_test,y_train,y_test,output_path):
    print("================TCN_svm_embedding Start=====================")
    clf = svm.SVC(kernel='rbf',C = 100,gamma=0.001)
    clf.fit(new_train,y_train)
    Y_pred = clf.predict(new_test)
    print(classification_report(y_test,Y_pred, digits=4))
    f2 = fbeta_score(y_test, Y_pred, beta=2)
    print("f2",f2)
    return Y_pred