# coding=utf-8
from sklearn import svm
from sklearn.metrics import classification_report
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer,fbeta_score

def svm_model(train_dataset,test_dataset,output_path):
    print("================SVM  Start=====================")
    ## 加载序列数据
    X_train = train_dataset[:, 0:13]
    Y_train = train_dataset[:, 13]
    X_test = test_dataset[:, 0:13]
    Y_test = test_dataset[:, 13]

    print("X_train.shape",X_train.shape)
    print("Y_train.shape",Y_train.shape)
    print("X_test.shape",X_test.shape)
    print("Y_test.shape",Y_test.shape)



    # parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},
    #               {'kernel': ['poly'], 'C': [1], 'degree': [2, 3]}]
    # parameters = {'gamma': [1e-3,1e-3, 1e-4],'C': [1, 10, 100]}
    # parameters = {'degree': [2, 3], 'C': [1, 10, 100]}
    # clf = GridSearchCV(svm.SVC(kernel='poly',gamma='scale'), parameters, cv=10, scoring=make_scorer(fbeta_score, beta=2))
    clf = svm.SVC(kernel='rbf',C = 100,gamma=0.001)
    clf.fit(X_train,Y_train)

    # print("Best parameters set found on development set:")
    # # 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
    # print(clf.best_params_)
    # print("Grid scores on development set:")
    # means = clf.cv_results_['mean_test_score']
    # stds = clf.cv_results_['std_test_score']
    #
    # # 看一下具体的参数间不同数值的组合后得到的分数是多少
    # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean, std * 2, params))
    # reg = clf.best_estimator_
    # # for key in parameters.keys():
    # #     print(key, reg.get_params()[key])
    # print("GS.best_params_", clf.best_params_)
    # print("GS.best_score_", clf.best_score_)

    #
    # score_rbf = reg.score(X_test,Y_test)
    # path = output_path + 'SVM_rbf'+str(score_rbf) + '.pkl'
    # joblib.dump(reg, path)
    # # joblib.dump(clf_rbf, '../output/SVM_rbf.pkl')
    # print("The score of rbf is : %f"%score_rbf)

    Y_pred = clf.predict(X_test)
    f2 = fbeta_score(Y_test, Y_pred, beta=2)
    print("f2",f2)
    print(classification_report(Y_test,Y_pred, digits=4))

    path = output_path + 'SVM-rbf'+str(f2)+'.pkl'
    joblib.dump(clf, path)
    return Y_pred
