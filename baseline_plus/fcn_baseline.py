# coding=utf-8

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import fbeta_score,classification_report
SEED = 6666666
time_steps, input_dim = 188, 1
nb_classes = 1
def fcn_model(test_dataset):
    print("================FCN Start=====================")
    dimen = len(test_dataset[0])
    print(dimen)
    x_test = test_dataset[:,0:dimen-1][:,:,np.newaxis,np.newaxis]
    y_test = test_dataset[:,dimen-1][:,np.newaxis]
    # # 加载旧模型
    model = load_model('../cpu_output/s120000r4/fcn0.8367768595041322.h5')
    model.summary()
    FCN_y_pred = model.predict(x_test).round()
    f2 = fbeta_score(y_test, FCN_y_pred, beta=2)
    print("f2",f2)
    print(classification_report(y_test, FCN_y_pred, digits=4))
    return FCN_y_pred