# coding=utf-8
SEED = 6666666

# 传递参数
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", help="model:KNN SVM DT TCN GRU", type=str, default='TCN')
parser.add_argument("-o", "--output", help="output path:", type=str, default='../cpu_output/s120000r4_baseline/')


args = parser.parse_args()
model_name = args.model_name
output_path = args.output
# print(output_path)
import os
if not os.path.exists(output_path):  # 如果路径不存在
    os.makedirs(output_path)

# gpu 减低随机性
os.environ['PYTHONHASHSEED']=str(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


import random
random.seed(SEED)

import numpy as np
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

# 导入各个模型
from knn_baseline import knn_model
from svm_baseline import svm_model
from dt_baseline import dt_model
from tcn_baseline import tcn_model
from fcn_baseline import fcn_model
from gru_baseline import gru_model
from cnn_baseline import cnn_model

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score,fbeta_score


dirlist = np.loadtxt('/home/wamdm/xinli/competition/baseline/matchedidlist_neg.csv',dtype=str,delimiter=",")
negative_test_dirlist = np.loadtxt('/home/wamdm/xinli/competition/gwac_gpu/dataset/023_15730595-G0013_negative_test_dirlist.csv',dtype=str,delimiter=",")
neg_predpositiveid = np.where(np.isin(negative_test_dirlist, dirlist) == True)[0]
# print(neg_predpositiveid)
equalseries_index_023 = np.loadtxt('/home/wamdm/xinli/competition/gwac_gpu/dataset/023_15730595-G0013_negative_test_equalseries_index.csv',dtype=int, delimiter=",")
testindex = np.where(np.isin(equalseries_index_023, neg_predpositiveid) == True)[0]
# print("testindex",testindex)
# 加载测试数据集
def load_test_data(model_name):
    # 加载全部的数据集
    if model_name == 'KNN' or model_name == 'TCN' or model_name == 'FCN' or model_name == 'Resnet' or model_name == 'GRU' or model_name == 'BIGRU' or model_name == 'CNN1D' or model_name == 'all':
        positive_dataset = np.loadtxt('../dataset/flare28_scale_dataset_label2.csv', dtype=float, delimiter=',')
        # negative_dataset = np.loadtxt('../dataset/baseline_neg_scale_dataset.csv', dtype=float, delimiter=',')
        negative_dataset = np.loadtxt('../dataset/023_15730595-G0013_negative_test.csv', dtype=float, delimiter=',')[testindex]
    elif model_name =='DT' or model_name =='MLP' or model_name =='GBDT' or model_name == 'SVM':
        positive_dataset = np.loadtxt('../dataset/flare28_scale_dataset_label2_feature.csv', dtype=float, delimiter=',')
        negative_dataset = np.loadtxt('../dataset/023_15730595-G0013_negative_test_feature.csv', dtype=float, delimiter=',')[testindex]
        # negative_dataset = np.loadtxt('../dataset/baseline_neg_scale_dataset_feature.csv', dtype=float, delimiter=',')

    test_dataset = np.vstack((negative_dataset,positive_dataset))
    print("test_dataset_size:")
    print("positive_dataset.shape = ",positive_dataset.shape)
    print("negative_dataset.shape = ",negative_dataset.shape)
    print("test_dataset.shape = ",test_dataset.shape)
    return test_dataset

# 回到id上计算各项评价指标
def metrics(pred,model_name):
    equalseries_index = np.loadtxt('../dataset/023_15730595-G0013_negative_test_equalseries_index.csv', dtype=int, delimiter=',')[testindex]
    # print(equalseries_index)
    flare_equalseries_index = np.loadtxt('../dataset/flare28_equalseries_index.csv', dtype=int, delimiter=',')
    pred = pred.astype(bool)
    np.set_printoptions(threshold=np.inf)
    finelpred = np.zeros(equalseries_index_023[-1] + 1 + flare_equalseries_index[-1] + 1, dtype='bool')

    flag = 0
    for j, pindex in enumerate(equalseries_index):
        if pindex == neg_predpositiveid[flag]:
            finelpred[neg_predpositiveid[flag]] = finelpred[neg_predpositiveid[flag]] | pred[j]
        else:
            print(finelpred[neg_predpositiveid[flag]])
            flag = flag + 1
            finelpred[neg_predpositiveid[flag]] = finelpred[neg_predpositiveid[flag]] | pred[j]

    flag = equalseries_index_023[-1] + 1
    for i, pindex in enumerate(flare_equalseries_index):
        if pindex + equalseries_index_023[-1] + 1 == flag:
            finelpred[flag] = finelpred[flag] | pred[i + len(equalseries_index)]
        else:
            flag = flag + 1
            finelpred[flag] = finelpred[flag] | pred[i + len(equalseries_index)]

    print("len(finelpred)", len(finelpred))
    finelpred = finelpred.astype(int)

    print("后来预测的正样本个数", len(np.where(finelpred == 1)[0]))
    # print(np.where(finelpred==1))
    np.savetxt(output_path + model_name + '_finelpred_onid.csv', finelpred, fmt='%d', delimiter=',')
    # 将baseline预测为负的样本数量考虑进来，还是在总的测试集上计算一个总的classification_report
    print("flare_pred_result:", finelpred[-(flare_equalseries_index[-1] + 1):len(finelpred)])

    # 真实id标签
    real_label = np.hstack((np.zeros(3976), np.ones(flare_equalseries_index[-1] + 1)))
    print("len(real_label)", len(real_label))

    print("classification_report on id")
    print(classification_report(real_label, finelpred, digits=4))
    print("f1", f1_score(real_label, finelpred))
    print("f2", fbeta_score(real_label, finelpred, beta=2))


# 调用各模型
if __name__ == '__main__':
    if model_name == 'DT':
        test_dataset = load_test_data('DT')
        pred_dt = dt_model(test_dataset)
        metrics(pred_dt,model_name)
        print("================DT END ============================")
    elif model_name =='KNN':
        test_dataset = load_test_data('KNN')
        pred_knn = knn_model(test_dataset)
        metrics(pred_knn,model_name)
        print("================KNN END ============================")
    elif model_name == 'SVM':
        test_dataset = load_test_data('SVM')
        pred_svm = svm_model(test_dataset)
        metrics(pred_svm,model_name)
        print("================SVM END ============================")
    elif model_name == 'TCN':
        test_dataset = load_test_data('TCN')
        pred_tcn = tcn_model(test_dataset)
        metrics(pred_tcn,model_name)
        print("================TCN END ============================")
    elif model_name == 'FCN':
        test_dataset = load_test_data('FCN')
        pred_fcn = fcn_model(test_dataset)
        metrics(pred_fcn,model_name)
        print("================FCN END ============================")
    elif model_name == 'GRU':
        test_dataset = load_test_data('GRU')
        pred_gru = gru_model(test_dataset)
        metrics(pred_gru,model_name)
        print("================GRU END ============================")
    elif model_name == 'CNN':
        pred_cnn = cnn_model(testindex)
        metrics(pred_cnn, model_name)
        print("================CNN END ============================")
