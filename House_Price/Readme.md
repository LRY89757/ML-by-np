# House Price

---

逯润雨 计卓2001

***为了能够使图片正常显示，本人将图片放到博客中了，所以难免会有水印。***

*由于本人写这一题之前已经将神经网络识别手写数字的代码完成，所以本题本人将神经网络直接套用到这里了，只不过改了一些误差损失函数，由交叉熵改为均方误差。*

神经网络的详解看这里:**[神经网络求解过程](readme about NN.md)**



## 最终结果

本人往kaggle里面提交了一共4次，最好结果是最后一次提交的，误差为0.16691，排在10440名

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210719104244871.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xSWTg5NzU3,size_16,color_FFFFFF,t_70)



![在这里插入图片描述](https://img-blog.csdnimg.cn/20210719104411188.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xSWTg5NzU3,size_16,color_FFFFFF,t_70)



**由于本人已经在[神经网络求解过程](readme about NN.md)里面介绍过神经网络部分的编写过程，所以本人打算介绍一下编写过程中遇到的困难以及相关解决方案。**



## 数据处理部分

个人参考了大量有关该数据的处理方案，也学到了许许多多类型的处理方式，也发现了特征工程画图处理确实非常方便， 本人参考很多数据分析之后决定为了便于神经网络的输入实现，将每一条输入数据变成331维的向量。数据处理过程如下：

1. 读取数据：

```python
# 读取数据
train_data = pd.read_csv('./house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('./house-prices-advanced-regression-techniques/test.csv')
```

2. 接着我们去除没有用的ID，然后连结所有样本特征：

```python
# 去除ID
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))   # 将所有的训练数据和测试数据的79个特征按样本连结。
```

3. 对连续数值的特征做标准化（standardization）：设该特征在整个数据集上的均值为 μ ，标准差为 σ 。那么，我们可以将该特征的每个值先减去 μ 再除以 σ 得到标准化后的每个特征值。对于缺失的特征值，我们将其替换成该特征的均值。标准化的目的是为了将我们的缺失数据填为0，且计算更加方便。

```python
# 标准化
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index   # 得到数值类型索引
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))     # 数值类型标准化方便填缺失数据            
# 标准化后，每个特征的均值变为0，所以可以直接用0来替换缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0)
```

4. 离散特征转化：将每一个离散型数据的分类都另起一种类别，是这一类的其值就为1反之为0，这样一来就将离散的特征转化为数字特征，从而方便我们数值处理，不过这样一来也增加了维度，特征从79个变为了331个。

```python
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape
```

5. 训练集、训练标签转化：将我们得到的特征转化为numpy的ndarray。方便进一步带入神经网络。

```python
# 上一步转换将特征数从79增加到了331。
# 最后，通过values属性得到NumPy格式的数据，并转成ndarray方便后面的训练。
n_train = train_data.shape[0]
train_features = np.array(all_features[:n_train].values)
test_features = np.array(all_features[n_train:].values)
train_labels = np.array(train_data.SalePrice.values).reshape((-1, 1))
```

6.   对于labels取log缩减值：我之前没有缩减数据规模，导致最终得到的均方损失函数的值溢出，几万的平方，确实会溢出，所以全部取log处理：

```python
# 预测标签集的数据使用log缩减，防止平方产生误差过大而溢出
train_labels = np.log(train_labels)
print(np.max(test_features))
print(np.max(train_features))
print(np.max(train_labels))


29.871533309269406
27.227905963254084
13.534473028231162
```

7. 划分测试集和训练集：

```python
# 划分训练时的测试集和训练集
train_ = np.random.choice(range(len(train_features)), 800, replace=False)
test_ = np.array([i for i in range(len(train_features)) if i not in train_])
train_features_ = train_features[train_]
train_labels_ = train_labels[train_]
test_features_ = train_features[test_]
test_labels_ = train_labels[test_]
```

8. 数据处理完毕

```python
# 至此我们数据处理完毕
print(test_features.shape, train_features.shape, train_labels.shape)
print(train_labels)

(1459, 331) (1460, 331) (1460, 1)
[[12.24769432]
 [12.10901093]
 [12.31716669]
 ...
 [12.49312952]
 [11.86446223]
 [11.90158345]]
```



## 神经网络改正部分

1. 误差损失函数：由于与识别手写数字的差别还是挺大，一个是回归，另一个是分类，我将最终的误差损失交叉熵改为了均方误差。

```python
# 误差损失函数定义（均方误差）
class MeanSquaredError:
    def __init__(self):
        '''
        Parameter:
        y : 预测输出结果矩阵，利用其求出误差损失，形状为(B, 1)
        label: 真实标签矩阵，形状为 (B， 1)
        '''
        self.loss = None
        self.z = None
        self.label = None
    
    # Mean Squared Error的前向传播
    def forward(self, y, label):
        '''
        input:
        y : 预测输出结果矩阵，利用其求出误差损失，形状为(B, 10)
        label: 真实标签矩阵，形状为 (B， 1)
        output:
        loss: 均方误差损失
        '''
        loss = 0.5 * np.sum((y - label) ** 2)
        self.loss = loss
        self.z = y
        self.label = label
        return loss
    
    # Mean Squared Error的反向传播
    def backward(self):
        '''
        output:
        均方误差梯度
        '''
        return self.z - self.label
```



2. 另外本人写了一个新的预测函数方便导出结果：

```python 
 # 预测函数
    def predict_(self, test_feature):
        
        x = test_feature
        y1 = self.layer_1.forward(x)  # 前向传播，一步步往后走
        z1 = self.activ_1.forward(y1)
        y2 = self.layer_2.forward(z1)
        z2 = self.activ_2.forward(y2)
        y3 = self.layer_last.forward(z2)
        return y3
```





## 代码调试

本人尝试了挺多次的神经网络神经元个数以及学习率，

![在这里插入图片描述](https://img-blog.csdnimg.cn/2021071911183297.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xSWTg5NzU3,size_16,color_FFFFFF,t_70)

最终得到最好的测试结果就是150、50的这一项，学习率为0.001，调了好长时间😭，这个代码的调试比手写数字难多了。最终提交到kaggle上也是这个数据集结果最好~~（仅仅对于我而言比较好了，1万名，感觉进一步需要数据处理上下功夫）~~



## 导出结果

最终导出结果，完结撒花！

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210719112240923.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xSWTg5NzU3,size_16,color_FFFFFF,t_70)

