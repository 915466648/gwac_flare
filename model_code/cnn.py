# coding=utf8

import numpy as np
import matplotlib.image as mpimg
SEED = 6666666
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam
from timeit import default_timer as timer
from sklearn.metrics import classification_report,fbeta_score
import os

# 将读取的img转化为矩阵
def img_to_array(imageFile):
    img = mpimg.imread(imageFile).astype(np.float)
    img=img[:,:,0]
    img=1-img
    return img.reshape((50,150,1))

def cnn_model(data_size,radio, output_path,batch,epoch,learnrate,iftemplate):
    print('================CNN Strat ============================')
    # 训练数据集
    train_negative_list = os.listdir('../dataset/train/train_negative/')
    train_negative_sample = random.sample(train_negative_list, int(radio / (radio + 1) * data_size))
    train_negative_pool = [os.path.join('../dataset/train/train_negative/',name) for name in train_negative_sample]
    if iftemplate == 1:
        train_positive_list = os.listdir('../dataset/train/train_positive_amplitude_left0.5/')
        print("==========template======")
        train_positive_sample = random.sample(train_positive_list, int(1 / (radio + 1) * data_size))
        train_positive_pool = [os.path.join('../dataset/train/train_positive_amplitude_left0.5/', name) for name in
                               train_positive_sample]
    else:
        train_positive_list = os.listdir('../dataset/train/augment_positive150000/')
        print("==========argument======")
        train_positive_sample = random.sample(train_positive_list, int(1 / (radio + 1) * data_size))
        train_positive_pool = [os.path.join('../dataset/train/augment_positive150000/', name) for name in train_positive_sample]

    train_name_list = np.hstack((train_negative_pool , train_positive_pool))
    y_train = np.vstack(([[0]]*len(train_negative_pool),[[1]]*len(train_positive_pool)))
    print("y_train.shape = ",y_train.shape)
    # 对训练集的数据和标签进行同步shuffle
    state = np.random.get_state()
    np.random.shuffle(train_name_list)
    np.random.set_state(state)
    np.random.shuffle(y_train)
    x_train = np.zeros((len(train_name_list), 50, 150, 1))
    for i,item in enumerate(train_name_list):
        x_train[i] = img_to_array(item)

    print("x_train.shape = ",x_train.shape)
    # 加载测试数据集，需要注意的是，需要按照文件名顺序读取，以便将片段对应到id上，并进一步计算在ID上的得分
    test_negative_list = [str(i)+'.png' for i in range(0,len(os.listdir('../dataset/test/test_negative/')))]
    test_negative_pool = [os.path.join('../dataset/test/test_negative/',name) for name in test_negative_list]

    test_positive_list = [str(i)+'.png' for i in range(0,len(os.listdir('../dataset/test/flare28_2/')))]
    # print(test_positive_list)
    test_positive_pool = [os.path.join('../dataset/test/flare28_2/',name) for name in test_positive_list]
    test_name_list = np.hstack((test_negative_pool, test_positive_pool))
    y_test = np.vstack(([[0]] * len(test_negative_pool), [[1]] * len(test_positive_pool)))

    x_test = np.zeros((len(test_name_list), 50, 150, 1))
    for i,item in enumerate(test_name_list):
        x_test[i] = img_to_array(item)

    print("train_dataset_size")
    print("x_train.shape = ",x_train.shape)
    print("x_test.shape = ",x_test.shape)

    # 模型
    input_shape = (50, 150, 1)   #（批、高度、宽度、通道）
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(5, 5), activation="relu",padding="same",kernel_initializer=glorot_uniform(seed=SEED),input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, kernel_size=(4, 4), activation="relu",padding="same",kernel_initializer=glorot_uniform(seed=SEED)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu",padding="same",kernel_initializer=glorot_uniform(seed=SEED)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation="relu",kernel_initializer=glorot_uniform(seed=SEED)))
    model.add(Dropout(0.5))
    model.add(Dense(84, activation="relu",kernel_initializer=glorot_uniform(seed=SEED)))
    model.add(Dropout(0.1))
    model.add(Dense(1, kernel_initializer=glorot_uniform(seed=SEED)))
    model.add(Dense(1, activation="sigmoid",kernel_initializer=glorot_uniform(seed=SEED)))
    model.summary()
    # model.optimizer.lr.assign(0.00001)
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr = learnrate), metrics=["binary_accuracy"])


    batch_size = batch
    epochs = epoch
    start_time = timer()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.25,shuffle=False)
    current_time = timer()
    print("train_time", (current_time - start_time) / epoch)


    # 评估模型
    # true_y = np.apply_along_axis(np.argmax, 1, y_test)
    pred_y = model.predict(x_test).round()
    f2 = fbeta_score(y_test, pred_y, beta=2)
    print("f2",f2)
    mp = output_path + "cnn"+str(f2)+".h5"
    model.save(mp)
    # pred_y = np.apply_along_axis(np.argmax, 1, pred_y)

    print(classification_report(y_test, pred_y, digits=4))
    return pred_y
