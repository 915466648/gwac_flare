# coding=utf-8

from tcn import tcn_full_summary
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import fbeta_score,classification_report
SEED = 6666666
batch_size, time_steps, input_dim = None, 188, 1

def tcn_model(test_dataset):
    print("================TCN Start=====================")
    dimen = len(test_dataset[0])
    print(dimen)
    x_test = test_dataset[:,0:dimen-1][:,:,np.newaxis]
    y_test = test_dataset[:,dimen-1][:,np.newaxis]
    # 加载旧模型
    my_tcn_model = load_model('../cpu_output/s120000r4/tcn_new_saved_model/tryrepeat95')
    ## my_tcn_model = load_model('../final_output/s120000r4_embedding/tcn_embedding0.829646017699115.h5')
    ## my_tcn_model.summary()
    tcn_full_summary(my_tcn_model, expand_residual_blocks=False)

    TCN_y_pred = my_tcn_model.predict(x_test).round()
    f2 = fbeta_score(y_test, TCN_y_pred, beta=2)
    print("f2", f2)
    print(classification_report(y_test, TCN_y_pred, digits=4))
    return TCN_y_pred