# coding=utf-8

# 在TCN 中，结合了序列和统计特征，看看效果
SEED = 6666666


# 传递参数
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", help="model:KNN SVM DT TCN GRU", type=str, default='TCN')
parser.add_argument("-s", "--data_size", help="datasize:", type=int, default=120000)
parser.add_argument("-r", "--radio", help="negative vs positive:", type=int, default=4)
parser.add_argument("-b", "--batch", help="batch_size:", type=int, default=512)
parser.add_argument("-e", "--epoch", help="epochs:", type=int, default=10)
parser.add_argument("-l", "--learn_rate", help="learning rate:", type=float, default=0.01)
parser.add_argument("-o", "--output", help="output path:", type=str, default='../cpu_output/s120000r4_embedding_tcnplus/')
parser.add_argument("-t", "--template", help="whether use template:", type=int, default=1)
parser.add_argument("-g", "--gpu", help="whether use gpu:", type=str, default='-1')

args = parser.parse_args()

model_name = args.model_name
data_size = args.data_size
radio = args.radio
batch = args.batch
epoch = args.epoch
learnrate = args.learn_rate
output_path = args.output
iftemplate = args.template
print("===========iftemplate:",iftemplate,"===========")
ifgpu = args.gpu
# print(output_path)
import os
if not os.path.exists(output_path):  # 如果路径不存在
    os.makedirs(output_path)

# gpu 减低随机性
os.environ['PYTHONHASHSEED']=str(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = ifgpu


import random
random.seed(SEED)

import numpy as np
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

from tcn_embedding import tcn_model
from tcn_embedding_gdbt import tcn_gdbt_model
from tcn_embedding_dt import tcn_dt_model
from tcn_embedding_svm import tcn_svm_model
from fcn_embedding import fcn_model
from resnet_embedding import resnet_model
from gru_embedding import gru_model
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score,fbeta_score
from tensorflow.keras.models import load_model,Model
from tcn import tcn_full_summary
from timeit import default_timer as timer


# TCN模型，将模板和普通增强的方式相结合
def load_train_data(model_name):
    # 加载全部的数据集
    if iftemplate == 1:
        positive_sequence = np.loadtxt('../dataset/044_16280425-G0013_scale_positive_dataset_amplitude_left0.5.csv', dtype=float, delimiter=',')
        positive_feature = np.loadtxt('../dataset/044_16280425-G0013_scale_positive_dataset_amplitude_left0.5_feature.csv', dtype=float,delimiter=',')
        print("==========template==========")
    else:
        positive_sequence = np.loadtxt('../dataset/augment_positive150000.csv', dtype=float, delimiter=',')
        positive_feature = np.loadtxt('../dataset/augment_positive_feature150000.csv', dtype=float, delimiter=',')
        print("==========argument==========")

    negative_sequence = np.loadtxt('../dataset/044_16280425-G0013_scale_negative_dataset.csv', dtype=float, delimiter=',')
    negative_feature = np.loadtxt('../dataset/044_16280425-G0013_scale_negative_dataset_feature.csv', dtype=float, delimiter=',')

    # 随机获得相应数量的正负样本，并拼接成一个，方便后续shuffle
    positive_sample_list = [i for i in range(len(positive_sequence))]
    pos_list= random.sample(positive_sample_list, int((1 / (radio + 1) * data_size)))
    positive_sequence_data = positive_sequence[pos_list, :]
    positive_feature_data = positive_feature[pos_list, :-1]
    positive_data = np.hstack((positive_feature_data,positive_sequence_data))

    negative_sample_list = [i for i in range(len(negative_sequence))]
    neg_list = random.sample(negative_sample_list, int(radio/(radio+1)*data_size))
    negative_sequence_data = negative_sequence[neg_list, :]
    negative_feature_data = negative_feature[neg_list, :-1]
    negative_data = np.hstack((negative_feature_data,negative_sequence_data))

    train_dataset = np.vstack((positive_data,negative_data))
    # np.random.seed(SEED)
    np.random.shuffle(train_dataset)
    print("train_dataset_size")
    print("positive_data.shape = ",positive_data.shape)
    # print(argument_data.shape)
    print("negative_data.shape = ",negative_data.shape)
    print("train&validate_dataset.shape = ",train_dataset.shape)
    return train_dataset

# 加载测试数据集
def load_test_data(model_name):
    # 加载全部的数据集
    positive_sequence = np.loadtxt('../dataset/flare28_scale_dataset_label2.csv', dtype=float, delimiter=',')
    negative_sequence = np.loadtxt('../dataset/023_15730595-G0013_negative_test.csv', dtype=float, delimiter=',')
    positive_feature = np.loadtxt('../dataset/flare28_scale_dataset_label2_feature.csv', dtype=float, delimiter=',')
    negative_feature = np.loadtxt('../dataset/023_15730595-G0013_negative_test_feature.csv', dtype=float, delimiter=',')

    positive_dataset = np.hstack((positive_feature[:,:-1], positive_sequence))
    negative_dataset = np.hstack((negative_feature[:,:-1], negative_sequence))
    test_dataset = np.vstack((negative_dataset,positive_dataset))
    print("test_dataset_size:")
    print("positive_dataset.shape = ",positive_dataset.shape)
    print("negative_dataset.shape = ",negative_dataset.shape)
    print("test_dataset.shape = ",test_dataset.shape)
    return test_dataset

# 回到id上计算各项评价指标
def metrics(pred,model_name):

    positive_index = np.where(pred == 1)[0]
    np.savetxt(output_path + model_name + '_positive_index.csv', positive_index, fmt='%d', delimiter=',')

    np.set_printoptions(threshold=np.inf)
    equalseries_index = np.loadtxt('../dataset/023_15730595-G0013_negative_test_equalseries_index.csv', dtype=int, delimiter=',')
    flare_equalseries_index = np.loadtxt('../dataset/flare28_equalseries_index.csv', dtype=int, delimiter=',')
    # 子序列层面上的预测，在pred中
    pred = pred.astype(bool)
    flag = 0
    # 以flag作为id的循环标签，将在子序列层面上的预测转化为ID层面上的预测，在finalpred中
    finelpred = np.zeros(equalseries_index[-1] + 1 +flare_equalseries_index[-1] + 1, dtype='bool')
    for i, pindex in enumerate(equalseries_index):
        if pindex == flag:
            finelpred[flag] = finelpred[flag] | pred[i]
        else:
            flag = flag + 1
            finelpred[flag] = finelpred[flag] | pred[i]

    print("len(finelpred)",len(finelpred))
    print(flag)
    flag = flag + 1
    for i, pindex in enumerate(flare_equalseries_index):
        if pindex + equalseries_index[-1] + 1 == flag:
            finelpred[flag] = finelpred[flag] | pred[i+len(equalseries_index)]
        else:
            flag = flag + 1
            finelpred[flag] = finelpred[flag] | pred[i+len(equalseries_index)]

    # print("len(finelpred)",len(finelpred))
    finelpred = finelpred.astype(int)
    np.savetxt(output_path + model_name + '_finelpred_onid.csv', finelpred, fmt='%d', delimiter=',')
    print("flare_pred_result:",finelpred[-(flare_equalseries_index[-1] + 1):len(finelpred)])
    # 真实id标签
    real_label =np.hstack((np.zeros(equalseries_index[-1]+1),np.ones(flare_equalseries_index[-1]+1)))
    print(classification_report(real_label, finelpred, digits=4))
    print("f1",f1_score(real_label, finelpred))
    print("f2",fbeta_score(real_label, finelpred, beta=2))

# 调用各模型
if __name__ == '__main__':
    train_dataset = load_train_data(model_name)
    test_dataset = load_test_data(model_name)
    dimen = len(train_dataset[0])
    print(dimen)
    x_train = train_dataset[:, 13:dimen - 1][:, :, np.newaxis]
    y_train = train_dataset[:, dimen - 1]
    x_test = test_dataset[:, 13:dimen - 1][:, :, np.newaxis]
    y_test = test_dataset[:, dimen - 1]

    print("x_train.shape", x_train.shape)
    print("y_train.shape", y_train.shape)
    print("x_test.shape", x_test.shape)
    print("y_test.shape", y_test.shape)

    # 加载旧模型
    my_tcn_model = load_model('../cpu_output/s120000r4/tcn_new_saved_model/tryrepeat95')
    ## my_tcn_model = load_model('../final_output/s120000r4_embedding/tcn_embedding0.829646017699115.h5')
    ## my_tcn_model.summary()
    tcn_full_summary(my_tcn_model, expand_residual_blocks=False)

    # 获得训练数据和测试数据序列在中间层的输出，
    # 整理训练数据是为了可以进一步训练下一个模型，输入到fit中，整理测试数据是为了在下一个模型中使用和测试。不整理成一样的格式就没法用模型
    middle = Model(inputs=my_tcn_model.input, outputs=my_tcn_model.get_layer('dense').output)
    middle_trian_result = middle.predict(x_train)
    middle_test_result = middle.predict(x_test)
    # 与特征拼接重新拼接训练数据
    new_train = np.hstack((middle_trian_result, train_dataset[:, 0:13]))
    new_test = np.hstack((middle_test_result, test_dataset[:, 0:13]))
    print(new_train.shape)

    if model_name =='TCN_GDBT':
        pred_tcn_gdbt = tcn_gdbt_model(new_train,new_test,y_train,y_test,output_path)
        metrics(pred_tcn_gdbt,model_name)
        print("================TCN_GDBT END ============================")
    if model_name =='TCN_DT':
        pred_tcn_dt = tcn_dt_model(new_train,new_test,y_train,y_test,output_path)
        metrics(pred_tcn_dt,model_name)
        print("================TCN_DT END ============================")
    if model_name =='TCN_SVM':
        pred_tcn_svm = tcn_svm_model(new_train,new_test,y_train,y_test,output_path)
        metrics(pred_tcn_svm,model_name)
        print("================TCN_SVM END ============================")
    if model_name =='all':
        pred_tcn_gdbt = tcn_gdbt_model(new_train, new_test, y_train, y_test, output_path)
        metrics(pred_tcn_gdbt, model_name)
        print("================TCN_GDBT END ============================")
        pred_tcn_dt = tcn_dt_model(new_train, new_test, y_train, y_test, output_path)
        metrics(pred_tcn_dt, model_name)
        print("================TCN_DT END ============================")
        pred_tcn_svm = tcn_svm_model(new_train, new_test, y_train, y_test, output_path)
        metrics(pred_tcn_svm, model_name)
        print("================TCN_SVM END ============================")




