import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score

real_neg = np.loadtxt('./023_15730595-G0013_negative_test_dirlist.csv',dtype=str,delimiter=",").tolist()
real_pos = np.loadtxt('./flare28_dirlist.csv',dtype=str,delimiter=",").tolist()
real_list = np.hstack((real_neg,real_pos))
real_label =np.hstack((np.zeros(len(real_neg)),np.ones(len(real_pos))))
# neg_pred = np.loadtxt('./matchedidlist_neg.csv',dtype=str,delimiter=",")
# print(len(neg_pred))
# negative_test_dirlist = np.loadtxt('/home/wamdm/xinli/competition/gwac_gpu/dataset/023_15730595-G0013_negative_test_dirlist.csv',dtype=str,delimiter=",")
# a_new = neg_pred
# for i in a_new:
#     if i not in negative_test_dirlist:
#         a_new = np.delete(a_new,np.where(a_new == i)[0][0])
# print("len(a_new)",len(a_new))
# print(a_new)
# 之前0.62的参数为/home/wamdm/competition_data/AstroSet-v0.1/AstroSet/023_15730595-G0013  abstar_template/flaretemplete.txt 20 200 2
pred_pos = np.hstack((np.loadtxt('./matchedidlist_neg.csv',dtype=str,delimiter=","),np.loadtxt('./matchedidlist_pos.csv',dtype=str,delimiter=",")))
finelpred = []
count = 0
for i,item in enumerate(real_list):
    if item in pred_pos:
        finelpred.append(1)
        count = count + 1
    else:
        finelpred.append(0)

print(finelpred[-28:len(finelpred)+1])
print('len(finelpred)',len(finelpred))
print('len(real_pos)',len(real_pos))
print('len(real_neg)',len(real_neg))
print('len(pred_pos)',len(pred_pos))

print("共预测正样本个数",count)
print(classification_report(real_label, finelpred, digits=4))
print("f1", f1_score(real_label, finelpred))
print("f2", fbeta_score(real_label, finelpred, beta=2))
