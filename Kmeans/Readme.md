# Kmeans聚类算法

---

逯润雨 计卓2001

***为了能够使图片正常显示，本人将图片放到博客中了，所以难免会有水印。***

## Kmeans算法代码分析

### 距离计算

这里计算的是样本数据集到每一个聚类中心的欧氏距离的平方，这里使用np.tile函数来使center向量来和data数据集向量可以做乘法减法，进而算出距离平方。

```python
# 计算某一聚类中中所有样本与样本中心的的欧氏距离的平方
def distance(data, center):
    '''
    input:data:
          center:样本中心向量
          data:某一聚类中中所有样本数据集(这里也可以是单个向量)
    output:distance(float):欧氏距离平方
    '''   
    if data.shape != center.shape:
        return np.sum((data - np.tile(center, (data.shape[0], 1))) * (data - np.tile(center, (data.shape[0], 1))), axis=1)
    else:
        return np.sum((data - center) * (data - center), axis=-1)
```

### Kmeans类各项参数

 k: 聚类的数目，默认为2    max_iterations: 最大迭代次数  edge: 判断是否收敛, 如果上一次的所有k个聚类中心与本次的所有k个聚类中心的差都小于edge，则说明算法已经收敛      flag:是否采用kmeans++算法优化初始值选取

*这里着重介绍一下flag，这个flag是选取初始化聚类中心的算法，默认为false，即普通类选取随机从样本集中选k个，如果为True那么就是采用kmeans++的方法初始化聚类中心*

```python
class Kmeans():
    """
    Parameters:
    k: 聚类的数目，默认为2
    max_iterations: int
        最大迭代次数.
    edge: float
        判断是否收敛, 如果上一次的所有k个聚类中心与本次的所有k个聚类中心的差都小于edge,
        则说明算法已经收敛
    flag:是否采用kmeans++算法优化初始值选取
    ans:分类结果
    centers:聚类中心集
    """

    def __init__(self, k=2, max_iterations=500, edge=0.0001, flag=False):
        self.k = k
        self.max_iterations = max_iterations
        self.edge = edge
        self.ans = {}  # 分类结果
        self.cnt = 1
        self.centers = None   # 聚类中心集
        self.flag = flag     # 是否采用kmeans++算法优化
```

### 初始化聚类中心

```python
# 从所有样本中随机选取self.k样本作为初始的聚类中心
    def init_centers(self, dataset):
        '''
       input:dataset:样本数据集
       output:centers:聚类中心的数据集
       '''
        dataset_nums = dataset.shape[0]  # 样本的数目
        centers = dataset[np.random.choice(np.arange(dataset_nums), self.k, replace=False)]   # 初始化聚类中心
        return centers
```

采用choice方法初始化聚类中心。

### Kmeans++算法初始化聚类中心

```python
    # 优化版选取初始值（kmeans++)
    def init_centersplus(self, dataset):
        '''
        input:dataset:样本数据集
        output:centers:聚类中心的数据集
        '''
        centers = []        # 聚类中心的数据集,最后会转化为ndarray
        dataset_nums = dataset.shape[0]       # 得到数据个数
        center = dataset[np.random.choice(dataset.shape[0], 1)][0]   # 随机得到一个数据集
        centers.append(center)                      # 加到聚类中心列表中
        for i in range(self.k - 1):
            min_dist = [np.min(distance(np.array(centers, dtype=object), sing_data))  for sing_data in dataset] # 得到每一个样本到聚点中心的最短距离(这里的距离是距离的平方)
            sum_dist = sum(min_dist)  # 得到总和
            p_dist = np.array(min_dist, dtype='float') / sum_dist  # 得到每一个的概率
            index = np.random.choice(range(dataset_nums), p=p_dist.ravel())  # 按照p_dist中的概率选出新聚类中心索引
            centers.append(dataset[index])  # 加到聚类中心集中
            
        return np.array(centers, dtype=object)  # 返回初始化结果
```

这里通过找每个样本离当前聚类中心最短距离的平方所占总平方和权重为概率来选择聚类中心，我们可以通过np.random.choice(p=)这一函数来按照一定概率选择初始化聚类中心。

### 求聚类分类

```python
    # 分类求在某聚类中心下每个点的分类
    def sort_class(self, centers, dataset):
        '''
        input:dataset:样本数据集
         centers:聚类中心的数据集
        output:每个样本的分类向量集
        '''
        dist = [distance(dataset, center) for center in centers]  # 用来装样本中每个点到各个聚类中心的距离
        dist_sort = np.argsort(dist, axis=0)  # 将其排序得到索引，此时dist_sort[0]就是每一个样本的分类类别
        dist_sorted = dist_sort[0]  # 每一个样本的分类类别向量集，向量里第i个元素的值j正好是第i个样本的种类类别j
        return dist_sorted
```

dist用来装样本中每个点到各个聚类中心的距离的平方，而dist_sort就是将其排序，这样第一行就是每一个样本的分类的类别。之后dist_sorted就是每一个样本的分类了。



### 计算聚类中心集合

```python
    # 计算聚类中心集合
    def cal_centers(self, dataset, dist_sorted):
        '''
        input:dataset:数据样本
        dist_sorted:每一个样本的分类类别向量集，每个向量里的值都是在dataset中的索引
        output:centers:数据样本聚类中心集合
        '''
        for i in range(self.k):
            self.ans[i] = []
        for i, vary in enumerate(dist_sorted):
            self.ans[vary].append(i)                # 更新每一个样本的分类
        
        return [np.sum(dataset[self.ans[i]], axis=0) / dataset[self.ans[i]].shape[0] for i in range(self.k) ]  # centers聚类中心集合
```

首先更新聚类分类结果（由上一步算出新聚类分类），而后根据np.sum(axis=0)得到我们新的聚类中心的集合（类型为ndarray)



### 判断是否收敛

```python

    # 判断是否收敛
    def judge_end(self, old_centers, centers):
        '''
        input:centers:新聚类中心集合
        old_centers:旧聚类中心集合
        output:True or False
        '''
        if self.cnt == 1:
            return False
        for old_center, center in zip(old_centers, centers):
            if np.sum(np.absolute(old_center - center)) > self.edge:
                return False
        return True
```

通过判断新旧聚类中心的重合程度来判断是否收敛



### 训练预测分类

```python
    # 预测分类聚类函数
    def predict(self, dataset):
        if self.flag == False:                   # 是否采用kmeans算法优化
            centers = self.init_centers(dataset)  # 初始化得聚类中心
        else:
            centers = self.init_centersplus(dataset)  # 初始化得聚类中心
#         print(centers)
        old_centers = None  # 旧的聚类中心
        
        while self.judge_end(old_centers, centers) == False and self.cnt <= self.max_iterations:   # 如果未到截止条件
            dist_sorted = self.sort_class(centers, dataset)    # 获得每一个样本的分类的矩阵
            old_centers = centers[::]                         # 旧的聚类中心的数组矩阵
            centers = self.cal_centers(dataset, dist_sorted)   # 新的聚类矩阵
            self.cnt += 1                               # 记录次数
        self.centers = centers                          # 记录聚类中心的数组矩阵
        return self.ans                                # 返回分类的字典，字典的键是每一类（0、1……k)，值是样本在dataset数据集中的索引
```

*注释写得非常详细了，几乎就是我们刚才讲解各种函数的那个顺序将代码实现一遍*



## Kmeans算法调试结果分析

```python
# 模型训练

k = int(input())
# centers = np.zeros((k, data.shape[-1]))

A = Kmeans(k=k, flag=True)
ans = A.predict(data)
centers = A.centers
print(centers)
print(A.ans)
```

输入不同k值来得到不同的聚类训练结果。



### 聚类不同k值图像展示

```python
# 画图
# for i in data:
#     plt.plot(i[0], i[1])
color = ['y', 'g', 'r', 'c', 'b']

# for i in range(k):
#     plt.plot(data[ans[i]][:, 0], data[ans[i]][:, 0], 'oc', markersize=0.8)
# for sing_data in data:
#     plt.plot(sing_data[0], sing_data[1], 'oc', markersize=0.8, c = 'y')
for i in range(k):            # 每一类的矩阵使用不同的颜色描述
    c = color[i % 5]
    for sing_data in data[ans[i]]:
        plt.plot(sing_data[0], sing_data[1], 'oc', markersize=0.8, c = c)



for center in centers:                              # 聚类中心使用红色五角星表示
    plt.plot(center[0], center[1], 'r*', markersize=16)
plt.title('Kmeans')
plt.box(False)
# xticks([])
# yticks([])
plt.show()
```



k = 2

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210719092943443.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xSWTg5NzU3,size_16,color_FFFFFF,t_70#pic_center)



k = 3:

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210719093200884.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xSWTg5NzU3,size_16,color_FFFFFF,t_70#pic_center)

k = 4:

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210719093142321.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xSWTg5NzU3,size_16,color_FFFFFF,t_70#pic_center)

k = 5

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210719093142327.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xSWTg5NzU3,size_16,color_FFFFFF,t_70#pic_center)



### 采用kmeans++优化后与kmeans对比分析、不同k值误差分析

```python
# 测试性能(不同K值/kmeans与kmeans++)

n = int(input())      # 重复训练次数
k = 15
x = list(range(2, k + 1))
y = []
yy = []
# centers = np.zeros((k, data.shape[-1]))
for k in range(2, k + 1):
    test_sum = 0.0
    for i in range(n):
        A = Kmeans(k=k, flag=True)
        ans = A.predict(data)
        centers = A.centers
    #     print(centers)
    #     print(A.ans)



        test = 0.0
        for i in range(k):
            test += sum(distance(data[ans[i]], centers[i]))**0.5
        
        test_sum += test / data.shape[0]         # 平均距离平方和
        #     print(test)
    
    y.append(test_sum / n)            

for k in range(2, k + 1):
    test_sum = 0.0
    for i in range(n):
        A = Kmeans(k=k, flag=False)
        ans = A.predict(data)
        centers = A.centers
    #     print(centers)
    #     print(A.ans)



        test = 0.0
        for i in range(k):
            test += sum(distance(data[ans[i]], centers[i]))**0.5
        
        test_sum += test / data.shape[0]         # 平均距离平方和
        #     print(test)
    
    yy.append(test_sum / n) 
    


plt.figure('the kmeans about k')
plt.title('Graph about k red->kmeans blue->kmeans++')
plt.box(False)
ax = plt.gca()
ax.set_xlabel('k')
ax.set_ylabel('loss')  #设置x轴、y轴名称
plt.plot(x, y, color='b', linewidth=1, alpha=0.6)
plt.plot(x, yy, color='r', linewidth=1, alpha=0.6)
plt.show()
# test
print(list(zip(x, y)))
print(list(zip(x, yy)))
```



得到图像分析为：图像横坐标代表k值，纵坐标代表损失误差Loss，红色的线为kmeans，蓝色为kmeans++。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210719093547423.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xSWTg5NzU3,size_16,color_FFFFFF,t_70#pic_center)



可以看出在k值较小的时候kmeans++与kmeans差异不大，但是k值变大之后kmeans++相对而言还是有一定优化的，虽然并不明显。



## 手写数字识别

经过代码套用分析发现，kmeans聚类判断效果并不好，个人认为是因为这里输入向量维度过高导致，输入784维度导致误差损失过大。我认为需要对输入进行降维处理。不过因为时间原因，本人并没有进一步探索处理。





