# coding=utf-8
# 调用PYPI 官方TCN包实现的TCN
# 引用
# https://pypi.org/project/keras-tcn/#implementation-results-1
from tcn import TCN, tcn_full_summary
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.metrics import fbeta_score,classification_report
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.initializers import glorot_uniform
SEED = 6666666
# if time_steps > tcn_layer.receptive_field, then we should not
# be able to solve this task.
batch_size, time_steps, input_dim = None, 188, 1

def tcn_model(train_dataset,test_dataset,output_path,batch,epoch,learnrate):
    print("================TCN Start=====================")
    dimen = len(train_dataset[0])
    print(dimen)
    x_train = train_dataset[:,0:dimen-1][:,:,np.newaxis]
    y_train = train_dataset[:,dimen-1][:,np.newaxis]
    x_test = test_dataset[:,0:dimen-1][:,:,np.newaxis]
    y_test = test_dataset[:,dimen-1][:,np.newaxis]

    filters = 16
    kernel_size = 4
    print("filters",filters,"kernel_size",kernel_size)
    tcn_layer = TCN(nb_filters=filters,dilations = (1,2,4,8,16,32),kernel_size = kernel_size,input_shape=(time_steps, input_dim),
                    return_sequences=False)

    # The receptive field tells you how far the model can see in terms of timesteps.
    print('Receptive field size =', int((tcn_layer.receptive_field-1)/2+1))

    my_tcn_model = Sequential()
    my_tcn_model.add(tcn_layer)
    # my_tcn_model.add(TCN(nb_filters=16,return_sequences=False))
    my_tcn_model.add(Dense(16, activation='relu', kernel_initializer=glorot_uniform(seed=SEED)))
    my_tcn_model.add(Dense(1, activation='sigmoid', kernel_initializer=glorot_uniform(seed=SEED)))

    my_tcn_model.compile(optimizer=Adam(lr=learnrate,decay=0.001), loss='binary_crossentropy', metrics=['binary_accuracy'])

    print(tcn_full_summary(my_tcn_model, expand_residual_blocks=False))
    # my_tcn_model.summary()
    # x, y = get_x_y()
    my_tcn_model.fit(x_train, y_train, epochs=epoch, validation_split=0.25,batch_size=batch,shuffle=False,verbose=1)
    TCN_y_pred = my_tcn_model.predict(x_test).round()
    f2 = fbeta_score(y_test, TCN_y_pred, beta=2)
    print("f2", f2)
    # mp = output_path + "tcn"+str(f2)+".h5"
    # my_tcn_model.save(mp)
    my_tcn_model.save(output_path +'tcn_new_saved_model/tryrepeat95')
    print(classification_report(y_test, TCN_y_pred, digits=4))
    return TCN_y_pred