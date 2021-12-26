# coding=utf8

import numpy as np
import matplotlib.image as mpimg
SEED = 6666666
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
import os

# 将读取的img转化为矩阵
def img_to_array(imageFile):
    img = mpimg.imread(imageFile).astype(np.float)
    img=img[:,:,0]
    img=1-img
    return img.reshape((50,150,1))

def cnn_model(testindex):
    print('================CNN Strat ============================')

    # 加载测试数据集，需要注意的是，需要按照文件名顺序读取，以便将片段对应到id上，并进一步计算在ID上的得分
    test_negative_list = [str(i)+'.png' for i in testindex]
    test_negative_pool = [os.path.join('../dataset/test/test_negative/',name) for name in test_negative_list]

    test_positive_list = [str(i)+'.png' for i in range(0,len(os.listdir('../dataset/test/flare28_2/')))]
    test_positive_pool = [os.path.join('../dataset/test/flare28_2/',name) for name in test_positive_list]

    test_name_list = np.hstack((test_negative_pool, test_positive_pool))
    y_test = np.vstack(([[0]] * len(test_negative_pool), [[1]] * len(test_positive_pool)))

    x_test = np.zeros((len(test_name_list), 50, 150, 1))
    for i,item in enumerate(test_name_list):
        x_test[i] = img_to_array(item)

    print("x_test.shape = ",x_test.shape)

    model = load_model('../cpu_output/s120000r4/cnn0.17834923268353378.h5')
    model.summary()
    pred_y = model.predict(x_test).round()
    print(classification_report(y_test, pred_y, digits=4))
    return pred_y
