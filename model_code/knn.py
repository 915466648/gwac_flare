# coding=utf-8
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import joblib
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer,fbeta_score
def f2_func(y_true, y_pred):
    f2_score = fbeta_score(y_true, y_pred, beta=2)
    return f2_score

def my_f2_scorer():
    return make_scorer(f2_func)


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

    parameters = {
        "n_neighbors": [*range(1, 10)]
    }
    knn = KNeighborsClassifier()
    GS = GridSearchCV(knn, parameters, cv=10, scoring=my_f2_scorer())  # cv交叉验证
    GS.fit(X_train, Y_train)

    print("Best parameters set found on development set:")
    # 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
    print(GS.best_params_)
    print("Grid scores on development set:")
    means = GS.cv_results_['mean_test_score']
    stds = GS.cv_results_['std_test_score']

    # 看一下具体的参数间不同数值的组合后得到的分数是多少
    for mean, std, params in zip(means, stds, GS.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    reg = GS.best_estimator_
    for key in parameters.keys():
        print(key, reg.get_params()[key])
    print("GS.best_params_",GS.best_params_)
    print("GS.best_score_",GS.best_score_)


    train_score_c = reg.score(X_train, Y_train)  # 返回预测的准确度
    test_score_c = reg.score(X_test, Y_test)
    print("train_score_c:", train_score_c, "test_score_c:", test_score_c)

    # path = output_path + 'KNN'+ str(test_score_c) + '.pkl'
    # joblib.dump(reg, path)

    # 输出预测性能指标
    y_pred = reg.predict(X_test)
    print(classification_report(Y_test,y_pred,digits=4))
    return y_pred


