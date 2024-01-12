import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, features, labels):
        self.features, self.means, self.stds = self.preprocessing_data(features)
        self.labels = np.array(labels)
        self.sample_num = self.features.shape[0]
        self.features_num = self.features.shape[1]
        self.weight = np.random.random((self.features_num, 1))
        self.losses = []

    def normalize_data(self, data):  # 数据标准化
        res = np.array(data)
        means = np.mean(res, axis=0)
        stds = np.std(res, axis=0)
        res = (res - means) / stds
        return res, means, stds

    def preprocessing_data(self, input):  # 数据标准化与偏置
        data, means, stds = self.normalize_data(input)
        data = np.column_stack((data, np.ones(data.shape[0])))
        return data, means, stds

    def preprocessing_test_data(self, test_data):  # 对测试数据进行标准化与添加偏置
        test_data_copy = np.array(test_data)
        test_data_copy = (test_data_copy - self.means) / self.stds
        test_data_copy = np.column_stack(
            (test_data_copy, np.ones(test_data_copy.shape[0]))
        )
        return test_data_copy

    def sigmoid(self, z):# sigmoid函数
        return 1 / (1 + np.exp(-z))

    def classify(self, data, threshold=0.5):# 根据sigmoid结果进行分类
        return np.where(data >= threshold, 1, 0)

    def cross_entropy_loss(self,prediction,labels):# 交叉熵损失函数
        ans = -np.mean(
            labels * np.log(prediction)
            + (1 - labels) * np.log(1 - prediction),
            axis=0,
        )
        return ans[0]

    def train(self, max_epoch,lr,batch=1,decay=0.1):# 随机梯度下降训练
        if batch>self.sample_num:
            raise ValueError("batch can not bigger than samle_num")
        learning_rate=lr
        for epoch in range(max_epoch):
            feature_selected=[]#随机样本矩阵
            label_selected=[]
            for i in range(batch):
                rand_index=np.random.randint(self.features.shape[0])#随机生成样本下标
                feature_selected.append(self.features[rand_index,:])
                label_selected.append(self.labels[rand_index,:])
                
            feature_selected=np.array(feature_selected)
            label_selected=np.array(label_selected)
            learning_rate=lr*1.0/(1.0+decay*epoch)
            prediction = self.sigmoid(feature_selected @ self.weight)
            gd = ((feature_selected.T @ (prediction - label_selected))) / feature_selected.shape[0]
            self.weight = self.weight - learning_rate * gd
            loss = self.cross_entropy_loss(prediction,label_selected)
            self.losses.append(loss)

    def predict(self, test_data):# 预测
        test_data = self.preprocessing_test_data(test_data)
        prediction = self.sigmoid((test_data @ self.weight))
        return prediction

    def compute_accuracy(self, fetures_test_data, labels_test_data):#计算准确率
        prediction = self.predict(fetures_test_data)
        prediction=self.classify(prediction)
        total = labels_test_data.shape[0]
        equal_count = 0
        for i in range(total):
            if prediction[i][0] == labels_test_data[i][0]:
                equal_count += 1
        return equal_count / total
