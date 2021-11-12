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


def cal_dtw_distance(ts_a, ts_b):
    """Returns the DTW similarity distance between two 2-D
    timeseries numpy arrays.
    Arguments
    ---------
    ts_a, ts_b : array of shape [n_samples, n_timepoints]
        Two arrays containing n_samples of timeseries data
        whose DTW distance between each sample of A and B
        will be compared
    d : DistanceMetric object (default = abs(x-y))
        the distance measure used for A_i - B_j in the
        DTW dynamic programming function
    Returns
    -------
    DTW distance between A and B
    """
    d = lambda x, y: abs(x - y)
    max_warping_window = 10000

    # Create cost matrix via broadcasting with large int
    ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    M, N = len(ts_a), len(ts_b)
    cost = sys.maxsize * np.ones((M, N))

    # Initialize the first row and column
    cost[0, 0] = d(ts_a[0], ts_b[0])
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + d(ts_a[i], ts_b[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j - 1] + d(ts_a[0], ts_b[j])

    # Populate rest of cost matrix within window
    for i in range(1, M):
        for j in range(max(1, i - max_warping_window),
                       min(N, i + max_warping_window)):
            choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

    # Return DTW distance given window
    return cost[-1, -1]



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

    # X_train, X_test, Y_train, Y_test = train_test_split(train_data,train_target,test_size=0.2, random_state=0)

    # 计算训练时间
    # neighbors = 2
    # weight = 'distance'
    # met = cal_dtw_distance
    # print("n_neighbors = ",neighbors, "weights = ",weight,"n_jobs = ",jobs)
    parameters = {
        # 'weights': ('distance', 'uniform'),
        # 'metric':('euclidean','manhattan','chebyshev'),
        "n_neighbors": [*range(1, 10)]
    }
    # knn = KNeighborsClassifier(n_neighbors = neighbors, weights = weight,n_jobs = -1)
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
    # knn.fit(X_train, Y_train)

    train_score_c = reg.score(X_train, Y_train)  # 返回预测的准确度
    test_score_c = reg.score(X_test, Y_test)
    print("train_score_c:", train_score_c, "test_score_c:", test_score_c)

    path = output_path + 'KNN'+ str(test_score_c) + '.pkl'
    joblib.dump(reg, path)

    # 输出预测性能指标
    y_pred = reg.predict(X_test)
    print(classification_report(Y_test,y_pred,digits=4))
    return y_pred

    # train_time = []
    # for i in range(10):
    #     start_time = timer()
    #     knn.fit(X_train, Y_train)
    #     current_time = timer()
    #     train_time.append(current_time-start_time)
    # #     print(current_time-start_time)
    #     i = i+1
    # print(train_time)
    # print(np.mean(train_time))
    # joblib.dump(knn, './output/knn.pkl')
    # knn = joblib.load('./output/knn.pkl')
    # # 计算测试时间
    # test_time = []
    # for i in range(10):
    #     start_time = timer()
    #     knn.predict(X_test)
    #     current_time = timer()
    #     test_time.append(current_time-start_time)
    # #     print(current_time-start_time)
    #     i = i+1
    # print(test_time)
    # print(np.mean(test_time))



