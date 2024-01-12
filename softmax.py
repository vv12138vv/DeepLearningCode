import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




class SoftmaxRegression:
    def __init__(self,features,labels):
        self.features, self.means, self.stds = self.preprocessing_data(features)
        self.labels = self.to_onehot(labels)
        self.sample_num = self.features.shape[0]
        self.features_num = self.features.shape[1]
        self.type_num=self.labels.shape[1]
        self.weight = np.random.random((self.features_num,self.type_num))
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
    
    def to_onehot(self,labels):
        type_num=max(labels)+1
        type_num=int(type_num[0])
        sample_num=labels.shape[0]
        onehot=np.zeros((sample_num,type_num))
        for i,cls in enumerate(labels):
            onehot[i,int(cls[0])]=1
        return onehot
    
    def softmax(self,x):
        res=[]
        for i in range(x.shape[0]):
            maxnum=max(x[i,:])
            x[i,:]=x[i,:]-maxnum
            t=np.exp(x[i,:])
            sum=np.sum(t)
            res.append(t/sum)
        return np.array(res)

    def softmax_loss(self,prediction):
        return -np.sum(self.labels*np.log(prediction))/self.sample_num
    
    def train(self,max_epoch,lr):
        for epoch in range(max_epoch):
            prediction=self.features @ self.weight
            prediction=self.softmax(prediction)
            g=-(self.features.T@(self.labels-prediction))/self.sample_num
            self.weight=self.weight-lr*g
            loss=self.softmax_loss(prediction)
            self.losses.append(loss)
            
    def softmax_classify(self,prediction):
        cls_res=np.zeros((self.sample_num,1))
        for i,row in enumerate(prediction):
            cls_res[i,0]=np.argmax(row)
        return cls_res
    
    def predict(self,test_data):
        test_data=self.preprocessing_test_data(test_data)
        prediction=self.softmax((test_data@self.weight))
        return prediction
    
    def compute_accuracy(self,features_test_data,labels_test_data):
        prediction=self.predict(features_test_data)
        prediction=self.softmax_classify(prediction)
        total=labels_test_data.shape[0]
        equal_count=0
        for i in range(total):
            if prediction[i][0]==labels_test_data[i][0]:
                equal_count+=1
        return equal_count/total
        