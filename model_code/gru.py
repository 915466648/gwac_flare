# coding=utf-8
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import GRU,Bidirectional
from timeit import default_timer as timer
# from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,fbeta_score
SEED = 6666666
from tensorflow.keras.initializers import glorot_uniform,orthogonal
from tensorflow.keras.optimizers import Adam,SGD

def gru_model(train_dataset,test_dataset,output_path,batch,epoch,learnrate):
    print("================GRU Start=====================")
    ## 加载序列数据
    time_steps = len(train_dataset[0]) - 1

    x_train = train_dataset[:,0:time_steps,np.newaxis]
    y_train = train_dataset[:,time_steps,np.newaxis]
    x_test = test_dataset[:,0:time_steps,np.newaxis]
    y_test = test_dataset[:,time_steps,np.newaxis]

    input_dim = x_train.shape[2]

    print("x_train.shape",x_train.shape)
    print("y_train.shape",y_train.shape)
    print("x_test.shape",x_test.shape)
    print("y_test.shape",y_test.shape)

    # 定义模型结构
    GRU_model = Sequential()
    GRU_model.add(GRU(units=64, activation='tanh',return_sequences=True,
                      kernel_initializer=glorot_uniform(seed=SEED), recurrent_initializer=orthogonal(seed=SEED),
                      input_shape=(time_steps, input_dim)))
    # GRU_model.add(GRU(units=128, activation='relu', return_sequences=True,kernel_initializer=glorot_uniform(seed=SEED),input_shape=(dimen-1, 1)))
    GRU_model.add(GRU(units=32, activation='tanh', return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED),recurrent_initializer=orthogonal(seed=SEED)))
    GRU_model.add(GRU(units=16, activation='tanh', return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED),recurrent_initializer=orthogonal(seed=SEED)))
    GRU_model.add(GRU(units=16, activation='tanh', return_sequences=False,kernel_initializer=glorot_uniform(seed=SEED),recurrent_initializer=orthogonal(seed=SEED)))
    GRU_model.add(Dense(16, activation='tanh', kernel_initializer=glorot_uniform(seed=SEED)))
    GRU_model.add(Dense(1, activation='sigmoid',kernel_initializer=glorot_uniform(seed=SEED)))

    GRU_model.compile(loss='binary_crossentropy', optimizer=Adam(lr = learnrate,decay = 0.001), metrics=['binary_accuracy'])
    GRU_model.summary()
    start_time = timer()
    GRU_model.fit(x_train, y_train, validation_split=0.25,batch_size=batch, epochs=epoch,shuffle = False)
    current_time = timer()
    print("train_time",(current_time-start_time)/epoch)

    # score = GRU_model.evaluate(x_test, y_test)
    # print("evaluate_score_c:", score)
    # # mp = output_path +"gru" +str(score[1])+".h5"
    # # GRU_model.save(mp)

    start_time = timer()
    gru_y_pred = GRU_model.predict(x_test).round()
    current_time = timer()
    print("test_time", current_time - start_time)
    f2 = fbeta_score(y_test, gru_y_pred, beta=2)
    print("f2", f2)
    mp = output_path + "gru" + str(f2) + ".h5"
    GRU_model.save(mp)
    print(classification_report(y_test, gru_y_pred, digits=4))
    return gru_y_pred

    # # 计算模型预测时间
    # test_time_gru = []
    # for i in range(10):
    #     start_time = timer()
    #     GRU_model.predict(x_test, batch_size=500)
    #     current_time = timer()
    #     test_time_gru.append(current_time-start_time)
    #     print(current_time-start_time)
    #     i = i+1
    # print(test_time_gru)
    # print(np.mean(test_time_gru))