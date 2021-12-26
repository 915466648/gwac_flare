# coding=utf-8

import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.metrics import fbeta_score,classification_report
from tensorflow.keras.layers import Input,Conv2D,Activation,Dense,BatchNormalization,GlobalAveragePooling2D
from tensorflow.keras.initializers import glorot_uniform
SEED = 6666666
# if time_steps > tcn_layer.receptive_field, then we should not
# be able to solve this task.
time_steps, input_dim = 188, 1
nb_classes = 1
def fcn_model(train_dataset,test_dataset,output_path,batch,epoch,learnrate):
    print("================FCN Start=====================")
    dimen = len(train_dataset[0])
    print(dimen)
    x_train = train_dataset[:,0:dimen-1][:,:,np.newaxis,np.newaxis]
    y_train = train_dataset[:,dimen-1][:,np.newaxis]
    x_test = test_dataset[:,0:dimen-1][:,:,np.newaxis,np.newaxis]
    y_test = test_dataset[:,dimen-1][:,np.newaxis]

    x = Input(x_train.shape[1:])
    #    drop_out = Dropout(0.2)(x)
    conv1 = Conv2D(128, 8, 1, padding='same',kernel_initializer=glorot_uniform(seed=SEED))(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    #    drop_out = Dropout(0.2)(conv1)
    conv2 = Conv2D(256, 5, 1, padding='same',kernel_initializer=glorot_uniform(seed=SEED))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    #    drop_out = Dropout(0.2)(conv2)
    conv3 = Conv2D(128, 3, 1, padding='same',kernel_initializer=glorot_uniform(seed=SEED))(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)

    full = GlobalAveragePooling2D()(conv3)
    out = Dense(nb_classes, activation='sigmoid',kernel_initializer=glorot_uniform(seed=SEED))(full)

    model = Model(inputs=x, outputs=out)
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=learnrate,decay=0.001),
                  metrics=['binary_accuracy'])

    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,patience=50, min_lr=0.0001)
    # monitor：要监测的数量。
    # factor：学习速率降低的因素。new_lr = lr * factor
    # patience：没有提升的epoch数，之后学习率将降低。
    # verbose：int。0：安静，1：更新消息。
    # mode：{auto，min，max}之一。在min模式下，当监测量停止下降时，lr将减少；在max模式下，当监测数量停止增加时，它将减少；在auto模式下，从监测数量的名称自动推断方向。
    # min_delta：对于测量新的最优化的阀值，仅关注重大变化。
    # cooldown：在学习速率被降低之后，重新恢复正常操作之前等待的epoch数量。
    # min_lr：学习率的下限。

    hist = model.fit(x_train, y_train, epochs=epoch, validation_split=0.25, batch_size=batch, shuffle=False, verbose=1)
    # ,callbacks=[reduce_lr]

    FCN_y_pred = model.predict(x_test).round()
    f2 = fbeta_score(y_test, FCN_y_pred, beta=2)
    print("f2", f2)
    model.save(output_path + "fcn"+str(f2)+".h5")

    print(classification_report(y_test, FCN_y_pred, digits=4))
    return FCN_y_pred