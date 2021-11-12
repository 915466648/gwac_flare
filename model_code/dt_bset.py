# coding=utf-8
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import numpy as np
import joblib
from sklearn.metrics import make_scorer,fbeta_score


def dt_model(train_dataset,test_dataset,output_path):
    print("================Decision Tree Start=====================")
    X_train = train_dataset[:, 0:13]
    Y_train = train_dataset[:, 13]
    X_test = test_dataset[:, 0:13]
    Y_test = test_dataset[:, 13]
    # X_train, X_test, Y_train, Y_test = train_test_split(train_data, train_target, test_size=0.2, random_state=0)
    print("X_train.shape",X_train.shape)
    print("Y_train.shape",Y_train.shape)
    print("X_test.shape",X_test.shape)
    print("Y_test.shape",Y_test.shape)

    ##  训练决策树
    features = 13
    depth = 8
    min_samples_leaf = 1
    splitter = 'best'
    criterion = 'entropy'
    # {'criterion': 'entropy', 'max_depth': 8, 'max_features': 13, 'min_samples_leaf': 1, 'splitter': 'best'}
    # print("max_features = ",features,"depth = ",depth)
    # parameters = {
    #     'splitter': ('best', 'random'),
    #      'criterion': ("gini", "entropy"),
    #      "max_depth": [*range(5, 10)],
    #      "max_features": [*range(5, 14)],
    #      'min_samples_leaf': [*range(1, 50, 5)]
    # }
    reg = DecisionTreeClassifier(max_features=features, max_depth=depth,criterion = criterion,min_samples_leaf = min_samples_leaf,splitter = splitter)
    # scoring = {'f2':make_scorer(fbeta_score, beta=2),'p':'precision','r':'recall'}
    # GS = GridSearchCV(DecisionTreeClassifier(), parameters,cv=10,scoring=make_scorer(fbeta_score, beta=2),return_train_score=True)  # cv交叉验证
    # # scoring :Strategy to evaluate the performance of the cross-validated model on the test set.
    reg.fit(X_train, Y_train)

    # print("Best parameters set found on development set:")
    # # 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
    # result = GS.cv_results_
    # print(GS.best_params_)
    # print("Grid scores on development set:")
    # means = GS.cv_results_['mean_test_score']
    # stds = GS.cv_results_['std_test_score']
    #
    # # 看一下具体的参数间不同数值的组合后得到的分数是多少
    # for mean, std, params in zip(means, stds, GS.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean, std * 2, params))
    #
    # reg = GS.best_estimator_
    # for key in parameters.keys():
    #     print(key, reg.get_params()[key])
    # print("GS.best_params_",GS.best_params_)
    # print("GS.best_score_",GS.best_score_)
    # # clf = clf.fit(X_train, Y_train)

    train_score_c = reg.score(X_train, Y_train)  # 返回预测的准确度
    test_score_c = reg.score(X_test, Y_test)
    print("train_score_c:",train_score_c, "test_score_c:",test_score_c)

    ### 进行决策树的模型剖析
    feature_name = ['beyond1std', 'kurtosis', 'skew', 'stetson_j', 'stetson_k', 'amp', 'max_slope', 'mad', 'fpr20',
                    'fpr35', 'fpr50', 'fpr65', 'fpr80']
    clf_importances = reg.feature_importances_
    clf_indices = np.argsort(clf_importances)[::-1]
    print("决策树的特征重要性排序")

    Y_pred = reg.predict(X_test)
    print(classification_report(Y_test, Y_pred, digits=4))

    path = output_path + 'DecisionTreeClassifier'+ str(test_score_c) + '.pkl'
    joblib.dump(reg, path)
    for f in range(0,13,1):
        print("%2d) %-*s %f" % (f + 1, 30, feature_name[clf_indices[f]], clf_importances[clf_indices[f]]))
    return Y_pred