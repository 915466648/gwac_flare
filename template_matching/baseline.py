# coding=utf8
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pywt
import _ucrdtw
import time
import os, sys
from fastdtw import fastdtw
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean, sqeuclidean
from thresholdpip import fastpip_threshold

def search_hdfs(fn, tmp_fn, len_min, len_max, threshold, start_index,output):
    anomaly_template_name = tmp_fn.split('/')[-1]
    anomaly_template = load_anomaly_template(tmp_fn)
    #fn_dir = fn.split('/')[:-1]   ##从位置0到位置-1之前的数  #从后往前数的话，最后一个位置为-1
    fnn = fn.split('/')[-1]
    result_dir=time.strftime('%Y-%m-%d',time.localtime(time.time()))+'@'
    num_sample = 1
    print(fn, num_sample)
    job_name = result_dir + anomaly_template_name + '.' + fnn + '.result'

    if not os.path.exists(job_name):
        os.mkdir(job_name)

    seg_num = 0
    seg_len_sum = 0
    density_sum = 0
    ret0 = []
    cal_count = 0
    valid_seg_len_sum = 0
    matched_count = 0
    matchidlist = []
    total_time_cost = 0
    base_path = fn
    files = os.listdir(base_path)
    files.sort()

    #delete abstract in files list
    if files[0]=='abstract':
        del files[0]

    for path in files:   ###处理一个数据文件
        full_path = os.path.join(base_path, path)
        with open(full_path) as fp:
            line = fp.read()
            line = line.split('\n')
            keys = ('JD', 'Magnorm', 'Magerror')
            data = {path: ''}
            values = []
            for items in line:
                items = items.split()
                dict = {k: v for k, v in zip(keys, items)}
                values.append(dict)
            data[path] = values
            for star in data:    ###我觉得有问题，每次读新的文件，data都是空的，但是此处又去遍历所有的星
                jd = []
                mag = []
                len_sampe = len(data[star])
                for i, item in enumerate(data[star]):  #i为序号，也可理解为行号，是enumerate的返回值，一行一行遍历
                    curr_jd = float(item['JD'])
                    tag_valid = not('sigma_ext_median' in item and 'sigmedthreshold' in item and float(item['sigma_ext_median']) > float(item['sigmedthreshold'])) and \
                                not('tag_valid' in item and float(item['tag_valid']) == 0)
#any() 函数用于判断给定的可迭代参数 iterable 是否全部为 False，则返回 False，如果有一个为 True，则返回 True。元素除了是 0、空、FALSE 外都算 TRUE。
                    if (any(jd) and curr_jd - jd[-1] > 0.25625 ) or len_sampe-1 == i:  ##如果间隔8个数据点或者是最后一行，则分片   间隔超过369分钟
                        # 0.0013889
                        seg_num += 1
                        if len_sampe-1 == i and tag_valid:  ###？？？
                            jd.append(curr_jd)
                            mag.append(float(item['Magnorm']))
                        l_mag = len(mag)     #the lenth
                        density_sum += cal_interval_density(jd)   ##？并不是片段密度，而是到目前为止所有的jd的密度
                        seg_len_sum += l_mag

                        if l_mag > 24:    #if l_mag>20, it is valid lenth
                            cal_count += 1
                            valid_seg_len_sum += l_mag
                            mag = np.array(mag)
                            jd = np.array(jd)
                            time_start = time.time()
                            # 传递一个子序列进行匹配
                            _ret0, matched, sim_min = \
                                        search_pattern_sigma_segment(jd, mag, anomaly_template, len_min, len_max, job_name + '/'+star, print_fig = True, threshold = threshold)
                            if matched:
                                matchidlist.append(star)
                            total_time_cost += time.time() - time_start
                            ret0.append(_ret0)
                            matched_count += matched

                        jd = []
                        mag = []
                    if tag_valid:
                        jd.append(curr_jd)
                        mag.append(float(item['Magnorm']))
    #output
    print('total time cost = ', total_time_cost)
    print('data_point_sum =', seg_len_sum)
    print("cal_count =", cal_count, "valid_seg_len_sum =", valid_seg_len_sum, 'avg =',valid_seg_len_sum / float(cal_count))
    print("matched_count =", matched_count)

    np.savetxt(output,list(set(matchidlist)),fmt="%s",delimiter=",")
    print("avg_density =", density_sum / float(seg_num))
    print("result saved in", job_name)

###加载模板文件
def load_anomaly_template(fn):
    raw_data = np.loadtxt(fn)    #raw_data.shape :(17,)    len(raw_data.shape):1
    if len(raw_data.shape) > 1:
        raw_data = raw_data[:, 1]
    raw_data = raw_data - raw_data[0]  #[ 0.   -1.1  -3.55 -3.15 -2.65 -2.35 -2.08 -1.87 -1.63 -1.35 -1.1  -0.9 -0.7  -0.38 -0.2   0.03  0.17]
    ##fit_transform先计算训练数据的均值和方差(fit)，
    # 还会基于计算出来的均值和方差来转换训练数据，从而把数据转换成标准的正态分布(transform)
    #-1被理解为unspecified value
    ###ret为60个点的模板y
    ret = StandardScaler().fit_transform(squeeze_tgt(raw_data, 60).reshape(-1, 1)).reshape(-1,)   #fit_transform先拟合数据，然后转化它将其转化为标准形式

    fig, axs = plt.subplots(1, 1, constrained_layout = True, figsize = (6,2))
    axs.invert_yaxis()
    axs.set_xlabel('N')
    axs.set_ylabel('Normalized Magnitude')  ###中间的那个图
    axs.plot(np.arange(ret.shape[0]), ret, '*', label='Query Pattern')
    plt.legend()  #加上图例
    plt.savefig('anomaly_template.eps')
    return ret

###计算片段数据的密度
def cal_interval_density(x):
    interval = 15 / 86400.0
    if len(x) < 2 or (x[-1] - x[0]) == 0:
        return 0

    overall_valid_percent = len(x)/ ((x[-1] - x[0]) / interval)
    return overall_valid_percent

###对模板数据进行了一维插值，返回y,其X为新的等差数组
def squeeze_tgt(ar, tgt_length):
    raw_len = len(ar)
    coef = float(raw_len) / float(tgt_length)   ###0.2833333
    #interp：一维线性插值.返回离散数据的一维分段线性插值结果.#https://blog.csdn.net/hfutdog/article/details/87386901
    # 与range 相比 arange返回一个array对象
    ##产生的数据点区间为[0,tgt_length* coef],步长为coef，如[0,20,40,....1180]
    #插值60个点
    return np.interp(np.arange(tgt_length) * coef , np.arange(raw_len), ar)

####search_pattern_sigma_segment(jd, mag, anomaly_template, len_min, len_max, job_name + '/'+star+'_'+str(curr_jd), print_fig = True, threshold = threshold)
def search_pattern_sigma_segment(timestamp, raw_time_series, ptn_scaled_ini, sig_len_min, sig_len_max, fn = '', print_fig = True, threshold = 0.4):
    # print("=============threshold==========",threshold)
    len_ts = raw_time_series.shape[0]   ###数据行数
    len_ptn = ptn_scaled_ini.shape[0]   ###模板行数

    cal_count = 0
    best_match_pattern_norm = None
    best_match_source_norm = None
    sim_min = np.inf
    best_match = None
    matched = False

    if len_ts <= sig_len_min or len_ts < 30:      ##如果片段长度小于参数中的20或者小于30，则丢弃
        return 0, matched, sim_min
    if sig_len_max == 0 or sig_len_max > len_ts:  ## 如果片段长度小于参数中的最大值，或者参数中的最大值为0，则重定义参数中的最大值为此片段的长度
        sig_len_max = len_ts
    if sig_len_min > len_ts:  ##如果片段的长度小于参数中的20，则把参数中的20重定义为片段长度
        sig_len_min = len_ts

    time_series = denoise_with_wavelets('sym6', 2, raw_time_series)
    noise_amp = cal_interval_denoise_variation(time_series, raw_time_series)
    pip_points, val = get_ts_pip_threshold(time_series, noise_amp)

    count = (sig_len_max - sig_len_min) * (len_ts - sig_len_max / 2)
    best_match_interval_deviation = 0
    best_match_substring_denoise_variation = 0
    max_interval_deviation = 0

    seg_max_list = []
    seg_min_list = []
    interval_denoise_variation_list = []

    for pip_start in range(len(pip_points) -1):
        start = pip_points[pip_start]
        end = pip_points[pip_start + 1]
        curr_ts = time_series[start : end].reshape(-1, 1)
        interval_denoise_variation_list.append(cal_interval_denoise_variation(curr_ts, raw_time_series[start : end].reshape(-1, 1)))
        seg_max_list.append(np.max(curr_ts))
        seg_min_list.append(np.min(curr_ts))

    for pip_start in range(len(pip_points) -1):
        for pip_end in range(pip_start + 1, len(pip_points)):
            start = pip_points[pip_start]
            end = pip_points[pip_end]
            substring_denoise_variation = np.mean(interval_denoise_variation_list[pip_start: pip_end+1])
            curr_ts = time_series[start : end].reshape(-1,)
            interval_deviation = curr_ts.max() - curr_ts.min()
            if interval_deviation > max_interval_deviation:
                max_interval_deviation = interval_deviation
            # 如果长度不满足条件，继续检查下一段
            if end - start >= sig_len_max or end - start < sig_len_min:
                continue
            #
            if interval_deviation < np.max([0.2, noise_amp *2]):
                continue
            if substring_denoise_variation > 1.4 * noise_amp or substring_denoise_variation > 0.4:#np.sum(ts_squared_error_sum[pip_start : pip_end]):
                continue

            ts_scaled = curr_ts.reshape(-1,)#scaler.transform(curr_ts).reshape(-1,)
            ptn_scaled = ptn_scaled_ini
            cal_count += 1
            ts_scaled = squeeze_tgt(ts_scaled, len_ptn).reshape(-1,)
            loc, distance = _ucrdtw.ucrdtw(ts_scaled, ptn_scaled, 0.05, False)
            distance = distance / ts_scaled.shape[0]
            sim = distance
            if sim < sim_min:
                sim_min = sim
                best_match_substring_denoise_variation = substring_denoise_variation
                best_match = (start, end)
                best_match_interval_deviation = interval_deviation
                best_match_source_norm = ts_scaled
                best_match_pattern_norm = ptn_scaled
    # 阈值越大，matched个数越多
    matched = (sim_min < threshold)
    if matched and print_fig:

        best_match_source_norm = StandardScaler().fit_transform(best_match_source_norm.reshape(-1, 1)).reshape(-1,)
        distance, best_match_path = fastdtw(best_match_source_norm, best_match_pattern_norm, dist = euclidean)
        print(distance / (best_match_pattern_norm.shape[0]), best_match_pattern_norm.shape[0])

        print('matched', best_match[0], best_match[1], 'sim ', sim_min)

        fig, axs = plt.subplots(3, 1, constrained_layout = False, figsize = (12,9))
        mse = mean_absolute_error(best_match_pattern_norm, best_match_source_norm)
        dtwd = sim_min / mse

        axs[0].invert_yaxis()
        axs[0].set_xlabel('JD')
        axs[0].set_ylabel('Magnitude')
        axs[0].plot(timestamp, time_series, '*')
        axs[0].plot(timestamp[best_match[0] : best_match[1]], time_series[best_match[0] : best_match[1]], color = 'red')


        axs[1].invert_yaxis()
        axs[1].set_xlabel('JD')
        axs[1].set_ylabel('Normalized Magnitude')

        print(best_match_path)

        for pair in best_match_path:
            axs[1].plot([pair[0],pair[1]], [best_match_source_norm[pair[0]], best_match_pattern_norm[pair[1]] ], '--', color = 'black')
        axs[1].plot(np.arange(best_match_source_norm.shape[0]), best_match_source_norm, '*-')
        axs[1].plot(np.arange(best_match_pattern_norm.shape[0]), best_match_pattern_norm, '.-')

        axs[2].invert_yaxis()
        axs[2].set_xlabel('JD')
        axs[2].set_ylabel('Magnitude')
        axs[2].plot(timestamp, raw_time_series, '*')
        axs[2].plot(timestamp, time_series, '--', color = 'yellow')
        axs[2].plot(timestamp[pip_points], val, '.-')

        fig.suptitle('Sim = ' + str(sim_min)[:5] + '/' + str(distance / (best_match[1] - best_match[0]))[:5]  + "DTWD=" + str(dtwd)[:5] + "  cal" + str(cal_count) + "/" + str(count) + " " + str(noise_amp)[:6] + '/' + str(best_match_interval_deviation)[:6] + '/' + str(best_match_substring_denoise_variation)[:6])
        flist = fn.split('/')
        out_fn = '/'.join(flist[:-1]) + '/' + str(sim_min)[:5] +'_' + flist[-1]
        plt.savefig(out_fn + '.png')

        plt.close()

    if matched:
        return best_match_source_norm, matched, sim_min
    return None, matched, sim_min

###计算去噪前后的差的中值，乘以5.76
def cal_interval_denoise_variation(ts, ts_denoised):
    diff = np.abs(ts - ts_denoised)
    return np.median(diff) * 5.76

###过滤高频成分，只保留第一个低频的近似分量，后面的高频的细节分量全部为0
def filter_high_freq_hard(coeffs, keep_level = 1):
    len_coeff = len(coeffs)
    for i in range(len_coeff - keep_level):
        coeffs[len_coeff-i-1] *= 0
    return coeffs

###小波变换去高频噪声
def denoise_with_wavelets(name = 'sym6', level = 2, time_series = None):
    wavelet_base = None
    try:
        wavelet_base = pywt.Wavelet(name)
    except Exception as e:
        pass
    ###pywt.wavedec此函数的返回值详解https://www.jianshu.com/p/9bad9466ad21
    coeffs = pywt.wavedec(time_series, wavelet_base, level = level)  #调用现成函数进行小波变换

    #过滤掉高频成分
    coeffs = filter_high_freq_hard(coeffs)
    #重建reconstruction
    time_series_hat = pywt.waverec(coeffs, wavelet_base)[:time_series.shape[0]]

    return time_series_hat

###节段化,返回分割点和分割点对应的y值，这个两个是分开的两个list
def get_ts_pip_threshold(time_series, threshold, x = None):
    if x is None:
        x = np.arange(time_series.shape[0])
    ###得到分割点的list
    pip_points = fastpip_threshold([(x[i], time_series[i]) for i in range(time_series.shape[0])], threshold)

    return pip_points, [time_series[i] for i in pip_points]

if __name__ == '__main__':
    start_index = 0
    threshold = 0.08   #相似度阈值
    len_min = 20
    len_max = 250
    tmp_fn =  './abstar_template/flare_star.txt'
    # tmp_fn = './abstar_template/flaretemplete.txt'

    #parameter list

    if(len(sys.argv) > 2):
        tmp_fn = sys.argv[2]   #abstar_templete
    if(len(sys.argv) > 3):
        len_min = int(sys.argv[3])
    if(len(sys.argv) > 4):
        len_max = int(sys.argv[4])
    if(len(sys.argv) > 5):
        threshold = float(sys.argv[5])
    if(len(sys.argv) > 1):
        fn = sys.argv[1]   #the file of light curve to be process
        print("fn:",fn)
    # /home/wamdm/competition_data/AstroSet-v0.1/AstroSet/023_15730595-G0013
    # /home/wamdm/xinli/competition/flare_dataset_31
    # /home/wamdm/xinli/competition/flare_dataset_28
    #process

    # 如果处理的是flare_dataset_28数据集，则将预测结果保存在matchedidlist1.csv
    # 如果处理的是023_15730595-G0013数据集，则将结果保存在matchedidlist2.csv
    # 这个需要手动改

    if fn =='/home/wamdm/competition_data/AstroSet-v0.1/AstroSet/023_15730595-G0013':
        output = './matchedidlist_neg.csv'
    elif fn =='/home/wamdm/xinli/competition/flare_dataset_28':
        output = './matchedidlist_pos.csv'
    else:
        output = './matchedidlist_3.csv'
    print("fn, tmp_fn, len_min, len_max, threshold, start_index", fn, tmp_fn, len_min, len_max, threshold, start_index,output)
    search_hdfs(fn, tmp_fn, len_min, len_max, threshold, start_index,output)
    # ref_032_06300085-G0013_1500724_9592
    # ref_043_04700255-G0013_29553_36981
    # ref_031_06300085-G0013_20382_64595
    exit()