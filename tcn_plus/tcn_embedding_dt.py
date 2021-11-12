# coding=utf-8

from sklearn.metrics import fbeta_score,classification_report
from sklearn.tree import DecisionTreeClassifier
SEED = 6666666
batch_size, time_steps, input_dim = None, 188, 1

def tcn_dt_model(new_train,new_test,y_train,y_test,output_path):
    print("================TCN_dt_embedding Start=====================")
    ##  训练决策树
    features = 29
    depth = 8
    min_samples_leaf = 1
    splitter = 'best'
    criterion = 'entropy'
    reg = DecisionTreeClassifier(max_features=features, max_depth=depth,criterion = criterion,min_samples_leaf = min_samples_leaf,splitter = splitter)
    reg.fit(new_train, y_train)
    Y_pred = reg.predict(new_test)
    print(classification_report(y_test, Y_pred, digits=4))
    f2 = fbeta_score(y_test, Y_pred, beta=2)
    print("f2",f2)
    return Y_pred