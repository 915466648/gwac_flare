# coding=utf-8
import numpy as np
from tensorflow.keras.models import load_model
from timeit import default_timer as timer
from sklearn.metrics import classification_report,fbeta_score
SEED = 6666666

def gru_model(test_dataset):
    print("================GRU Start=====================")
    ## 加载序列数据
    time_steps = len(test_dataset[0]) - 1
    x_test = test_dataset[:,0:time_steps,np.newaxis]
    y_test = test_dataset[:,time_steps,np.newaxis]

    print("x_test.shape",x_test.shape)
    print("y_test.shape",y_test.shape)

    GRU_model = load_model('../cpu_output/s120000r4/gru0.7877461706783371.h5')
    GRU_model.summary()
    start_time = timer()
    gru_y_pred = GRU_model.predict(x_test).round()
    current_time = timer()
    print("test_time", current_time - start_time)
    f2 = fbeta_score(y_test, gru_y_pred, beta=2)
    print("f2", f2)
    print(classification_report(y_test, gru_y_pred, digits=4))
    return gru_y_pred