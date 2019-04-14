# GBDT
multiple-classification GBDT model on dataset "20 newsgroups"

数据集：http://qwone.com/~jason/20Newsgroups/20news-19997.tar.gz

代码被分为两个部分，data_processing.py用于加载数据集、划分训练/测试集，以及清洗数据；model.py用于将文本转化为数值特征（TFIDF），特征筛选（卡方），训练模型，以及在测试集上评估模型的效果。另有parameter.py用于对GBDT模型进行调参。
