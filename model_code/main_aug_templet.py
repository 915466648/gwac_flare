# coding=utf-8
SEED = 6666666

# 取一半的augment 和 一半的templet

# 传递参数
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", help="model:KNN SVM DT TCN GRU", type=str, default='TCN')
parser.add_argument("-s", "--data_size", help="datasize:", type=int, default=120000)
parser.add_argument("-r", "--radio", help="negative vs positive:", type=int, default=4)
parser.add_argument("-b", "--batch", help="batch_size:", type=int, default=512)
parser.add_argument("-e", "--epoch", help="epochs:", type=int, default=10)
parser.add_argument("-l", "--learn_rate", help="learning rate:", type=float, default=0.01)
parser.add_argument("-o", "--output", help="output path:", type=str, default='../cpu_output/s120000r4_augment_templet/')
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
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ifgpu

# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# os.environ['HOROVOD_FUSION_THRESHOLD']='0'

import random
random.seed(SEED)

import numpy as np
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

# # from tfdeterminism import patch
# # patch()
# tf.compat.v1.set_random_seed(SEED)
# from keras import backend as K
# # session_conf = tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=True)
# session_conf = tf.compat.v1.ConfigProto()
# # session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存
# ######### session_conf.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# K.set_session(sess)
# tf.compat.v1.disable_eager_execution()
# 导入各个模型
from knn_best import knn_model
from svm import svm_model
from dt_bset import dt_model
from mlp import mlp_model
from tcn_new import tcn_model
from fcn import fcn_model
from resnet import resnet_model
from gru import gru_model
# from bigru import bigru_model
from cnn import cnn_model
from gbdt import gbdt_model
# from cov1D import cnn1d_model

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score,fbeta_score
from timeit import default_timer as timer


# 加载所有的正负样本，然后根据总数据量和正负比例的参数，从中取一部分
# ！！！！！！！！！！！！！一半模板生成的，一半是数据增强的！！！！！！！
def load_train_data(model_name):
    # 加载全部的数据集
    if model_name == 'KNN' or model_name == 'TCN' or model_name == 'FCN' or model_name == 'Resnet' or model_name == 'GRU' or model_name == 'all':
        positive_template = np.loadtxt('../dataset/044_16280425-G0013_scale_positive_dataset_amplitude_left0.5.csv', dtype=float, delimiter=',')
        positive_augment = np.loadtxt('../dataset/augment_positive150000.csv', dtype=float, delimiter=',')
        negative_dataset = np.loadtxt('../dataset/044_16280425-G0013_scale_negative_dataset.csv', dtype=float, delimiter=',')
    elif model_name =='DT' or model_name =='MLP' or model_name =='GBDT' or model_name == 'SVM':
        positive_template = np.loadtxt('../dataset/044_16280425-G0013_scale_positive_dataset_amplitude_left0.5_feature.csv', dtype=float, delimiter=',')
        positive_augment = np.loadtxt('../dataset/augment_positive_feature150000.csv', dtype=float, delimiter=',')
        negative_dataset = np.loadtxt('../dataset/044_16280425-G0013_scale_negative_dataset_feature.csv', dtype=float, delimiter=',')

    # 随机获得相应数量的正负样本，并拼接成一个，方便后续shuffle
    positive_sample_list = [i for i in range(len(positive_augment))]
    # argument_sample_list = [i for i in range(len(argument_dataset))]
    # random.seed(SEED)
    positive_template_data = positive_template[random.sample(positive_sample_list, int((1 / (radio + 1) * data_size)/2)), :]
    positive_augment_data = positive_augment[random.sample(positive_sample_list, int((1 / (radio + 1) * data_size)/2)), :]
    positive_data = np.vstack((positive_template_data,positive_augment_data))
    negative_sample_list = [i for i in range(len(negative_dataset))]
    # random.seed(SEED)
    negative_data = negative_dataset[random.sample(negative_sample_list, int(radio/(radio+1)*data_size)), :]

    train_dataset = np.vstack((positive_data,negative_data))
    # np.random.seed(SEED)
    np.random.shuffle(train_dataset)
    print("train_dataset_size")
    print("positive_template_data.shape = ", positive_template_data.shape)
    print("positive_augment_data.shape = ", positive_augment_data.shape)
    print("positive_data.shape = ",positive_data.shape)
    # print(argument_data.shape)
    print("negative_data.shape = ",negative_data.shape)
    print("train&validate_dataset.shape = ",train_dataset.shape)
    return train_dataset

# 加载测试数据集
def load_test_data(model_name):
    # 加载全部的数据集
    if model_name == 'KNN' or model_name == 'TCN' or model_name == 'FCN' or model_name == 'Resnet' or model_name == 'GRU' or model_name == 'BIGRU' or model_name == 'CNN1D' or model_name == 'all':
        positive_dataset = np.loadtxt('../dataset/flare28_scale_dataset_label2.csv', dtype=float, delimiter=',')
        negative_dataset = np.loadtxt('../dataset/023_15730595-G0013_negative_test.csv', dtype=float, delimiter=',')
    elif model_name =='DT' or model_name =='MLP' or model_name =='GBDT' or model_name == 'SVM':
        positive_dataset = np.loadtxt('../dataset/flare28_scale_dataset_label2_feature.csv', dtype=float, delimiter=',')
        negative_dataset = np.loadtxt('../dataset/023_15730595-G0013_negative_test_feature.csv', dtype=float, delimiter=',')


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
    pred = pred.astype(bool)
    flag = 0
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
    start_time = timer()
    if model_name == 'CNN':
        pred_cnn = cnn_model(data_size, radio, output_path,batch,epoch,learnrate,iftemplate)
        metrics(pred_cnn,model_name)
        print("================CNN END ============================")
    elif model_name == 'DT':
        train_dataset = load_train_data('DT')
        test_dataset = load_test_data('DT')
        pred_dt = dt_model(train_dataset, test_dataset, output_path)
        metrics(pred_dt,model_name)
        print("================DT END ============================")
    elif model_name == 'GBDT':
        train_dataset = load_train_data('GBDT')
        test_dataset = load_test_data('GBDT')
        pred_gbdt = gbdt_model(train_dataset, test_dataset, output_path)
        metrics(pred_gbdt,model_name)
        print("================GBDT END ============================")
    elif model_name == 'MLP':
        train_dataset = load_train_data('MLP')
        test_dataset = load_test_data('MLP')
        pred_mlp = mlp_model(train_dataset, test_dataset, output_path, batch, epoch, learnrate)
        metrics(pred_mlp,model_name)
        print("================MLP END ============================")
    else:
        train_dataset = load_train_data(model_name)
        test_dataset = load_test_data(model_name)
        if model_name =='KNN':
            pred_knn = knn_model(train_dataset,test_dataset,output_path)
            metrics(pred_knn,model_name)
            print("================KNN END ============================")
        elif model_name == 'SVM':
            pred_svm = svm_model(train_dataset,test_dataset,output_path)
            metrics(pred_svm,model_name)
            print("================SVM END ============================")
        elif model_name == 'TCN':
            pred_tcn = tcn_model(train_dataset,test_dataset,output_path,batch,epoch,learnrate)
            metrics(pred_tcn,model_name)
            print("================TCN END ============================")
        elif model_name == 'FCN':
            pred_fcn = fcn_model(train_dataset,test_dataset,output_path,batch,epoch,learnrate)
            metrics(pred_fcn,model_name)
            print("================FCN END ============================")
        elif model_name == 'Resnet':
            pred_resnet = resnet_model(train_dataset,test_dataset,output_path,batch,epoch,learnrate)
            metrics(pred_resnet,model_name)
            print("================Resnet END ============================")
        elif model_name == 'GRU':
            pred_gru = gru_model(train_dataset,test_dataset,output_path,batch,epoch,learnrate)
            metrics(pred_gru,model_name)
            print("================GRU END ============================")
        elif model_name =='all':
            pred_knn = knn_model(train_dataset,test_dataset,output_path)
            metrics(pred_knn,model_name)
            print("================KNN END ============================")
            pred_svm = svm_model(train_dataset,test_dataset,output_path)
            metrics(pred_svm,model_name)
            print("================SVM END ============================")
            pred_tcn = tcn_model(train_dataset,test_dataset,output_path,batch,epoch,learnrate)
            metrics(pred_tcn,model_name)
            print("================TCN END ============================")
            pred_gru = gru_model(train_dataset,test_dataset,output_path,batch,epoch,learnrate)
            metrics(pred_gru,model_name)
            print("================GRU END ============================")
            train_dataset = load_train_data('DT')
            test_dataset = load_test_data('DT')
            pred_dt = dt_model(train_dataset,test_dataset,output_path)
            metrics(pred_dt,model_name)
            print("================DT END ============================")
            pred_mlp = mlp_model(train_dataset,test_dataset,output_path,batch,epoch,learnrate)
            metrics(pred_mlp,model_name)
            print("================MLP END ============================")
            pred_cnn = cnn_model(data_size, radio, output_path,batch,epoch,learnrate)
            metrics(pred_cnn,model_name)
            print("================CNN END ============================")
            end_time = timer()
            print("time_consume:",end_time-start_time)
        elif model_name =='all_gpu':
            pred_tcn = tcn_model(train_dataset,test_dataset,output_path,batch,epoch,learnrate)
            metrics(pred_tcn,model_name)
            print("================TCN END ============================")

            train_dataset = load_train_data('MLP')
            test_dataset = load_test_data('MLP')
            pred_mlp = mlp_model(train_dataset,test_dataset,output_path,batch,epoch,learnrate)
            metrics(pred_mlp,model_name)
            print("================MLP END ============================")

            pred_cnn = cnn_model(data_size, radio, output_path,batch,epoch,learnrate)
            metrics(pred_cnn,model_name)
            print("================CNN END ============================")

            end_time = timer()
            print("time_consume:",end_time-start_time)
