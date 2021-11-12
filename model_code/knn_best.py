# coding=utf-8
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import joblib
from sklearn.metrics import fbeta_score

def knn_model(train_dataset,test_dataset,output_path):
    print("================KNN Start=====================")
    ## 加载序列数据
    dimen = len(train_dataset[0])
    #
    X_train = train_dataset[:,0:dimen-1]
    Y_train = train_dataset[:,dimen-1]
    X_test = test_dataset[:,0:dimen-1]
    Y_test = test_dataset[:, dimen-1]

    print("X_train.shape",X_train.shape)
    print("Y_train.shape",Y_train.shape)
    print("X_test.shape",X_test.shape)
    print("Y_test.shape",Y_test.shape)

    # 计算训练时间
    neighbors = 2
    knn = KNeighborsClassifier(n_neighbors = neighbors)
    knn.fit(X_train, Y_train)
    # 输出预测性能指标
    y_pred = knn.predict(X_test)
    f2 = fbeta_score(Y_test,y_pred,beta=2)
    path = output_path + 'KNN'+ str(f2) + '.pkl'
    joblib.dump(knn, path)
    print(classification_report(Y_test,y_pred,digits=4))
    return y_pred



