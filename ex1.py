from cmath import exp # 导入指数的数学函数
import random # 导入随机函数
import pandas as pd # 导入pandas库，用于处理数据集
import numpy as np # 导入numpy库，用于矩阵运算
import matplotlib.pyplot as plt

# 定义函数，读取数据集，输入参数为文件名，返回类型为ndarray的属性、分类二维表。
def Get_Data(name:str):
    return pd.read_csv(name).to_numpy()[:,1:]

# 定义二分类对数几率回归模型类
class Logistic_Regression:
    def __init__(self) -> None: # 构造函数
        self.X=None # 训练用数据矩阵
        self.Y=None # 数据真实值矩阵
        self.ω=None # 特征权重矩阵
        self.b=7 # 常数项
        self.max_iterations=0 # 最大迭代次数
        self.α=0 # 学习率
        self.θ=0 # 损失接受限

    # 设置模型参数的函数，包括学习率α、迭代次数上限，损失接受限θ
    def Set_Parameter(self, α:float=0.01, iterations:int=4000, θ=1e-2):
        self.α=α
        self.max_iterations=iterations
        self.θ=θ

    # 设置训练所需数据的函数
    def Set_Data(self, X, Y, ω):
        self.X=X
        self.Y=Y
        self.ω=ω

    # 设置测试集的函数
    def Set_Test(self,test_data):
        self.test=test_data

    # 计算损失函数值的函数，采用2-范数定义。
    def loss(self):
        return np.linalg.norm(self.Y-self.Y_predict)/len(self.Y)

    # sigmoid函数，用于将预测值转换为0~1之间的概率值。
    def sigmoid(self,X:np.ndarray)->float:
        #print(X) 
        #print(self.ω)
        return 1/(1+np.exp(-np.dot(X, self.ω)-self.b))   


    # 训练函数，使用梯度下降方法更新模型权重。
    def train(self):
        # print(self.X);print(self.ω)
        for i in range(self.max_iterations):
            self.Y_predict=self.sigmoid(self.X) # 计算训练集预测值
            # 使用梯度下降更新权重，其中np.transpose()表示转置运算，np.linalg.norm()表示求矩阵2-范数。
            self.ω-=self.α*np.dot(np.transpose(self.X),(self.Y_predict-self.Y))/len(self.X)
            self.b-=self.α*np.linalg.norm(np.dot(np.transpose(self.X),(self.Y_predict-self.Y)))/len(self.X) 
            #print(self.loss())  
            #print(self.θ)  
            
            # 当损失函数值小于阈值θ时，停止迭代。
            if(self.loss()<self.θ): 
                #print("self.loss<self.θ")
                break

    # 预测函数，根据模型和输入的新数据，输出0~1之间的概率值。
    def predict(self,X): 
        return self.sigmoid(X)

# 定义多分类OVR模型的类
class OVR:
    def __init__(self) -> None:
        self.data=None # 数据集
        self.LR=Logistic_Regression() # 二分类对数几率回归模型
        self.types=[] # 类型
        self.models=[] # 模型

    # 设置数据集的函数，根据输入的名称读取相应的数据集文件。
    def Set_DataSet(self,name):
        while (True):
            if (name == "西瓜"):
                self.data = Get_Data("watermelon.csv")
                break
            elif (name == "鸢尾花"):
                self.data = Get_Data("iris.csv")
                break
            else:
                print("无此数据集，请重新输入")
                break


    # 将数据集分成训练集和测试集的函数，输入参数为训练集比例train_ratio（0~1）。
    def data_split(self,train_ratio=0.75):
        self.types=list(set(self.data[:,-1])) # 将所有类型移到一个列表中
        data_train = [] # 空列表，用于存放训练所需数据
        self.data_test = [] # 空列表，用于存放测试所需数据

        # 将原始数据集随机划分为训练集和测试集
        for i in range(self.data.shape[0]):
            if (random.random() < train_ratio): # random.random()用于生成0~1之间的浮点数
                data_train.append(list(self.data[i]))
            else:
                self.data_test.append(list(self.data[i]))
        data_train=np.array(data_train)
        self.data_test=np.array(self.data_test)

        # 将数据集中的每个类型转化为0/1，构建OVR模型
        for k in range(len(self.types)):
            MODEL=[]
            for j in range(7):#多次同样模型，避免误差极端性
                # 初始化二分类对数几率回归模型，并将数据中的每个类型转化为0/1。
                model=Logistic_Regression() 
                model.Set_Parameter() # 设置模型参数
                data_train_copy = data_train.copy()
                for x in data_train_copy:
                    if(x[-1]==self.types[k]): x[-1]=1
                    else: x[-1]=0
                model.Set_Data(data_train_copy[:,:-1].astype(np.float32),data_train_copy[:,-1].astype(np.float32),np.random.rand(data_train.shape[1]-1))
                MODEL.append(model)
            self.models.append(MODEL)

    # 在训练集上训练所有OVR模型
    def train(self):
        for i in range(len(self.models)):
            for model in self.models[i]:
                model.train()

    # 在测试集上测试OVR模型
    def test(self):
        a=len(self.data_test) # 测试集数据量
        b=0 # 预测正确个数
        for x in self.data_test:
            SCORES=[]
            for i in range(len(self.models)):
                scores=[]
                for model in self.models[i]:
                    scores.append(model.predict(x[:-1].astype(np.float32)))
                SCORES.append(np.average(scores))
                print(SCORES)

            
            # 找到预测得分最高的类型，与真实类型比较
            if(self.types[SCORES.index(max(SCORES))]==x[-1]): b+=1 
        # 打印统计结果，包括训练集大小、测试集大小、预测正确数及准确率。
        print("有%d个数据用于训练，%d个数据用于测试，预测正确%d个，正确率%.2f%%"%(len(self.data)-a,a,b,b/a*100))


# 主程序
if __name__=='__main__':
    name=input("请输入所需数据集:") # 输入数据集名称
    for i in range(1,8):    # 输出了7个结果，在数据集中的不同比例下测试模型的准确性。
        ovr=OVR() # 初始化OVR模型
        ovr.Set_DataSet(name) # 设置数据集
        ovr.data_split(i/8.0) # 将数据集划分为训练集和测试集
        ovr.train() # 在训练集上训练OVR模型
        ovr.test() # 在测试集上测试OVR模型


        
        '''plt.xlabel('epoch')
        plt.ylabel('delta_weight')
        plt.plot(range(len(ch)), ch)
        plt.legend(['watermelon dataset3.0a'])
        plt.show()  '''