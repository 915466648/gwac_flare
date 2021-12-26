# coding=utf-8
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import joblib

def dt_model(train_dataset,test_dataset,output_path):
    print("================Decision Tree Start=====================")
    X_train = train_dataset[:, 0:13]
    Y_train = train_dataset[:, 13]
    X_test = test_dataset[:, 0:13]
    Y_test = test_dataset[:, 13]

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

    reg = DecisionTreeClassifier(max_features=features, max_depth=depth,criterion = criterion,min_samples_leaf = min_samples_leaf,splitter = splitter)
    reg.fit(X_train, Y_train)

    train_score_c = reg.score(X_train, Y_train)  # 返回预测的准确度
    test_score_c = reg.score(X_test, Y_test)
    print("train_score_c:",train_score_c, "test_score_c:",test_score_c)

    # ### 进行决策树的模型剖析
    # feature_name = ['beyond1std', 'kurtosis', 'skew', 'stetson_j', 'stetson_k', 'amp', 'max_slope', 'mad', 'fpr20',
    #                 'fpr35', 'fpr50', 'fpr65', 'fpr80']
    # clf_importances = reg.feature_importances_
    # clf_indices = np.argsort(clf_importances)[::-1]
    # print("决策树的特征重要性排序")
    # for f in range(0, 13, 1):
    #     print("%2d) %-*s %f" % (f + 1, 30, feature_name[clf_indices[f]], clf_importances[clf_indices[f]]))

    Y_pred = reg.predict(X_test)
    print(classification_report(Y_test, Y_pred, digits=4))

    path = output_path + 'DecisionTreeClassifier'+ str(test_score_c) + '.pkl'
    joblib.dump(reg, path)

    return Y_pred