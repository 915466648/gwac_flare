## 代码结构

------

- #### data_process：数据预处理代码。


​			|——diff-023_15730595-G0013.ipynb：对GWAC原始数据中的023_15730595-G0013的数据进行处理，包括截断、降噪、标准化，记录生成的每一个子序列所属的ID，作为测试集的一部分。

​			|——diff-044_16280425-G0013.ipynb：对GWAC原始数据中的023_15730595-G0013的数据进行处理，包括截断、降噪、标准化，记录生成的每一个子序列所属的ID，并利用耀发模型公式的方式生成耀发信号，并注入部分负样本中。即利用文中第二种数据增强方式生成均衡数据集。

​			|——diff-044_16280425-G0013_augment.ipynb：将利用少量耀发信号生成的正样本注入部分负样本中，即利用文中第一种数据增强方式生成均衡训练集。

​			|——diff-flare28.ipynb：对已知耀发数据进行处理，包括截断、降噪、标准化，记录生成的每一个子序列所属的ID，作为测试集的一部分。

​			|——feature.py：计算SVM、Decision Tree所需的13个特征。

​			|——flare12_new.ipynb：为文中第一种数据增强方式所需的样本做预处理，将经过处理的耀发片段保存。

​			|——plot4flare28.ipynb：绘制CNN模型所需的图片数据。通过更改输入文件，为所有数据生成

------

- #### dataset：部分处理后的数据。由于文件过大，因此无法将所有处理后的数据上传。读者可使用数据预处理方法与原数据自行处理。

​			|——023_15730595-G0013_negative_test.csv：测试集中的负样本

​			|——flare28_scale_dataset_label2.csv：测试集中的正样本。

​			|——044_16280425-G0013_scale_negative_dataset.csv：训练集中的负样本

​			|——044_16280425-G0013_scale_positive_dataset_amplitude_left0.5.csv：训练集中的正样本，此正样本通过注入耀发模型的方式生成。		

​			|——augment_positive150000.csv：训练集中的正样本，此正样本通过注入由少量真实耀发子序列生成的耀发信号生成，即第二种数据增强方式。

- ------

-   #### model_code ：基本的模型用法，其中包括网格搜索的过程。



​			|——fcncam.ipynb：绘制FCN模型的CAM热力图

​			|——cnn.py：CNN模型的训练及测试函数，并保存最佳模型。

​			|——dt.py：Decision Tree模型的网格搜索函数。

​			|——dt_best.py：使用Decision Tree模型的最佳参数配置进行训练及测试的函数，并保存最佳模型。

​			|——fcn.py：FCN模型的训练及测试函数，并保存最佳模型。

​			|——gru.py：GRU模型的训练及测试函数，并保存最佳模型。

​			|——knn.py：KNN模型的网格搜索函数。

​			|——knn_best.py：使用KNN模型的最佳参数配置进行训练及测试的函数，并保存最佳模型。

​			|——main_run.py：主函数，加载数据、计算评价指标、传递参数、调用各子函数。

​			|——svm.py：使用SVM模型的最佳参数配置进行训练及测试的函数，并保存最佳模型。（网格搜索过程在注释中）

​			|——tcn_new.py：TCN模型的训练及测试函数，并保存最佳模型。

- ------

-   #### cpu_output/s120000r4，保存的最佳模型，后续tcn_plus和baseline_plus需要使用



​			|——tryrepeat95：TCN的最佳模型

​			|——DecisionTreeClassifier0.995464991573464.pkl：DecisionTree的最佳模型

​			|——SVM-rbf0.666058394160584.pkl：SVM的最佳模型

​			|——cnn0.17834923268353378.h5：CNN的最佳模型

​			|——fcn0.8367768595041322.h5：FCN的最佳模型

​			|——gru0.7877461706783371.h5：GRU的最佳模型

​			|——KNN0.2606461086637298.pkl：KNN的最佳模型

- ------

-   #### tcn_plus：将TCN提取的特征与人工提取的特征进行结合，共同输入到SVM和Decision Tree中。



​			|——main_embedding_tcn.py：主函数，加载数据、计算评价指标、传递参数、调用各子函数。

​			|——tcn_embedding_dt.py：将TCN提取的特征与人工提取的特征进行结合，输入到Decision Tree模型进行训练和预测的函数。

​			|——tcn_embedding_svm.py：将TCN提取的特征与人工提取的特征进行结合，输入到SVM模型进行训练和预测的函数。

- ------

-   #### baseline_plus：级联方式，包括baseline级联其他模型和SVM与Decision Tree级联其他模型



​			|——cnn_baseline.py：利用最佳CNN模型将template_matching或SVM或Decision Tree初筛过的样本进一步筛选的子函数。

​			|——dt_baseline.py：利用最佳Decision Tree模型将template_matching或SVM初筛过的样本进一步筛选的子函数。

​			|——fcn_baseline.py：利用最佳FCN模型将template_matching或SVM或Decision Tree初筛过的样本进一步筛选的子函数。

​			|——gru_baseline.py：利用最佳GRU模型将template_matching或SVM或Decision Tree初筛过的样本进一步筛选的子函数。

​			|——knn_baseline.py：利用最佳KNN模型将template_matching或SVM或Decision Tree初筛过的样本进一步筛选的子函数。

​			|——main_baseline.py：主函数，加载template_matching初筛后的数据并调用相应的子函数进一步分类。

​			|——main_dt.py：主函数，加载Decision Tree初筛后的数据并调用相应的子函数进一步分类。

​			|——main_svm_final.py：主函数，加载SVM初筛后的数据并调用相应的子函数进一步分类。

​			|——svm_baseline.py：利用最佳SVM模型将template_matching或Decision Tree初筛过的样本进一步筛选的子函数。

​			|——tcn_baseline.py：利用最佳TCN模型将template_matching或SVM或Decision Tree初筛过的样本进一步筛选的子函数。

- ------

- #### template_matching：科学领域传统的经典方法，模板匹配



- ### README.md


- #### requirments.txt：本项目所必须的包及版本

  GWAC数据地址：https://tianchi.aliyun.com/dataset/dataDetail?dataId=88856

  Baseline（Template Matching）代码地址：https://tianchi.aliyun.com/competition/entrance/531805/introduction




