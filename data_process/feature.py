# coding=utf-8
from sklearn import preprocessing
from numpy import loadtxt, sum, size, where, abs, average, std, ones,sign, sqrt, max, min, diff, sort, floor, array, concatenate
import numpy as np
from scipy import stats
from timeit import default_timer as timer

# 要先对序列打标签，然后才能计算特征！！！！！！！否则本代码不适用
def calculate_F(n, m, mag):
    flux = 10 ** (-0.4 * mag)
    N = size(flux)
    sorted_flux = sort(flux)
    n = int(floor(N * n / 100))
    m = int(floor(N * m / 100))
    f_n = sorted_flux[n]
    f_m = sorted_flux[m]
    return f_m - f_n

def moment_based_features(mag):
    n = size(mag)
    ave_mag = average(mag)
    weights = 1
    wtd_ave_mag = average(mag)

    delta = std(mag)
    beyond1std = size(where(abs(mag - wtd_ave_mag) > delta)) / n
    kurtosis = stats.kurtosis(mag)
    skew = stats.skew(mag)

    delta_i = sqrt(n / (n - 1)) * ((mag - ave_mag) / delta)
    P_k = delta_i ** 2 - 1
    stetson_j = sum(weights * sign(P_k) * sqrt(abs(P_k))) / sum(weights)
    stetson_k = 1 / n * sum(abs(delta_i)) / sqrt(1 / n * sum(delta_i ** 2))

    return [beyond1std, kurtosis, skew,stetson_j, stetson_k]

def magnitude_based_features(mag):
    slope = diff(mag)
    amp = max(mag) - min(mag)
    max_slope = max(abs(slope))
    mad = stats.median_abs_deviation(mag)

    return [amp, max_slope, mad]


def percentile_based_features(mag):
    F_5_95 = calculate_F(5, 95, mag)
    fpr20 = calculate_F(40, 60, mag) / F_5_95
    fpr35 = calculate_F(32.5, 67.5, mag) / F_5_95
    fpr50 = calculate_F(25, 75, mag) / F_5_95
    fpr65 = calculate_F(17.5, 82.5, mag) / F_5_95
    fpr80 = calculate_F(10, 90, mag) / F_5_95

    return [fpr20, fpr35, fpr50, fpr65, fpr80]


def get_feature(mag):
    mag_scaled = preprocessing.scale(mag)
    features1 = moment_based_features(mag_scaled)
    features2 = magnitude_based_features(mag_scaled)
    features3 = percentile_based_features(mag_scaled)
    return array(features1 + features2 + features3)


# 得到提取的特征，依次为beyond1std, kurtosis, skew, stetson_j, stetson_k，amp, max_slope, mad，fpr20, fpr35, fpr50, fpr65, fpr80
def get_data(path):
    magarray = loadtxt(path,dtype=float,delimiter = ',')
    feature = ones([len(magarray), 14])
    dimen = len(magarray[0])
    print(dimen)
    for i,maglist in enumerate(magarray):
        mag = maglist[0:dimen-1]
        feature[i, :] = np.hstack((get_feature(mag) ,maglist[dimen-1:dimen]))
    return feature


if __name__ == '__main__':
    # positive = get_data('../dataset/flare28_scale_label_dataset.csv')
    # np.savetxt('../dataset/flare28_scale_label_dataset_feature.csv', positive ,fmt='%f',delimiter=',')
    positive = get_data('../dataset/augment_positive.csv')
    np.savetxt('../dataset/augment_positive_feature.csv', positive, fmt='%f', delimiter=',')
    # negative = get_data('../dataset/044_16280425-G0013_scale_negative_dataset.csv')
    # np.savetxt('../dataset/044_16280425-G0013_scale_negative_feature.csv', negative ,fmt='%f',delimiter=',')
