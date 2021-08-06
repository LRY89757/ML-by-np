

## 神经网络求解过程

---

*本人比较习惯记笔记，所以基本推导都记在笔记本里了……但是文中也会有latex公式辅助理解*



首先学习了有关神经网络的主要背景、了解了一些最基本的模型比如所MP模型，感知器、多层感知器，也看了看有关吴恩达、李宏毅的有关课程，~~虽然觉得看视频太慢了~~。大致了解了有关神经网络的大致组成轮廓。

### 前向传播

#### 矩阵表示

这些是之前就差不多知道的一些东西。但是一直不知道如何用数学语言实现， 而后通过参考一些书籍学到了如何使用矩阵描述所谓参数相乘，也即前向传播过程，本人笔记如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210715221148913.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xSWTg5NzU3,size_16,color_FFFFFF,t_70#pic_center)

#### 全连接层的前向传播代码

写到这里说明一下有关前向传播的代码，我定义了一个有关全连接层的类，里面包含了该层前向传播、后向传播的代码，定义如下(这里先将反向传播删去了）：

```python
# 全连接层的初始化类
class FullyConnected():
    def __init__(self, W, b):
        '''
        Parameter:
        W: 权重矩阵，形状为(N, M), N为输入神经元的个数，M为输出神经元的个数
        b: 偏移量矩阵 (M，)
        '''
        self.W = W    # 赋值
        self.b = b    # 赋值

        self.x = None      # 用来存储输入神经元的矩阵，为反向传播提供便利

        self.dW = None   # 用来存储梯度，梯度下降时需要用来更新权重
        self.db = None
```



全连接层类的初始化就是我们的权重矩阵W，偏移量矩阵b，我们假设 N为输入神经元的个数， B为输入神经元批量大小，M为输出神经元的个数，那么W的形状就是(N， M)， b形状就是（M，），b可以借助广播机制得到加和。而前向传播的过程也十分简单，就是依据的笔记中的公式Y=$$X^T$$W + b.不过numpy的形状和矩阵是略有不同的，所以这一点小心就是。不算特别难(*上下两个代码块是衔接在一起的*)：

```python

    # 全连接层的前向传播
    def forward(self, x):
        '''
        input:
        x: 输入神经元的矩阵, 形状(B, N)，B为批量大小，N为输入神经元的个数
        output:
        y: 输出神经元的矩阵, 形状(B, M)， M为输出神经元的个数
        '''
        self.x = x            # 存储输入神经元的矩阵，便于反向传播计算更新权重
        out = np.dot(self.x, self.W) + self.b   # 完成一次前向传播
        return out    # 返回前向传播结果
```



#### 激活函数的前向传播代码

当然这只是我们的全连接层权重W、b线性组合的前向传播，接下来是我们的有关激活函数的前向传播，由于激活函数的前向传播只是对每一个全连接层得到的矩阵中每一个元素进行一下有关函数运算，矩阵本身形状没有改变，所以我本来以为也不会很困难，事实上激活函数我也各自写了一个类，以下是Sigmoid、Relu函数前向传播部分的有关代码，这里relu函数就借鉴了有关博客的代码，个人认为写的很好，我也成功将其用在反向传播里面了：

```python
    # Sigmoid激活函数的前向传播。
    def forward(self, y):
        '''
        input: 
        y:全连接层前向传播得到矩阵，形状为(B, N)
        output:
        z:激活函数作用后得到矩阵，形状为(B, N)
        '''
        z = np.exp(y) / (1 + np.exp(y)) # 利用np.exp直接对矩阵运算
        self.z = z   # 赋值
        return z  # 返回矩阵
```

```python
    # Relu激活函数的前向传播。
    def forward(self, y):
        '''
        input: 
        y:全连接层前向传播得到矩阵，形状为(B, N)
        output:
        z:激活函数作用后得到矩阵，形状为(B, N)
        '''
        self.mask = (y <= 0)   # 得到关于y大于小于0的真值的矩阵
        z = y.copy()       # 深度拷贝一个y矩阵
        z[self.mask] = 0   # 将小于零的值赋为0
        return z   # 返回矩阵
```

###### 前向传播解决溢出问题

当然，本来以为这样就很简单完成了，但是后来发现运行时程序会报错，程序提示exp函数会溢出，我也找了许许多多有关博客后来写了一个相对很丑陋的代码（分输入值大于0和小于0处理），但是之后数据处理的时候发现了如果将数据归一化，把mnist数据集的像素值除以255，使其值不超过1的话，就很轻松的解决了exp溢出的问题。

### 反向传播的推导

接下来就是重头戏反向传播了，反向传播确实对于代码、具体数学实现了解不多，本次也正好借此机会学一学，反向传播实际上就是**求误差函数梯度，依据链式法则**，而求梯度的目的便是**更新权重参数以达到使误差函数尽可能小，使模型尽可能接近真实值**， 主要为了更新的就是我们的W、b这两项参数。求解过程我们分类来看，以sigmoid为激活函数

##### 单层感知器反向传播：

以E作为误差损失函数：

![在这里插入图片描述](https://img-blog.csdnimg.cn/2021071523050950.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xSWTg5NzU3,size_16,color_FFFFFF,t_70#pic_center)

##### 多层感知器反向传播：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210715230659959.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xSWTg5NzU3,size_16,color_FFFFFF,t_70#pic_center)

##### sigmoid缺陷

由此我们也可以看出sigmoid函数如果参数远小于0或远大于0的话求导的结果都很接近0，这么一来使得梯度接近于0，无法调整权重，所以我们训练过程中需要调整学习率防止梯度消失。

##### 反向传播矩阵运算处理

因为我们的权重这些都是使用的矩阵表示，所以我们如果可以得到对于矩阵的梯度反向传播，那是最好的，而实际上对于矩阵的求导也是有固定规则的，比如对一个全连接层来说，我们只需要使用矩阵求导规则，然后利用Y = $$X^T$$W + b这个公式就可以：（~~实际上也可以根据矩阵的形状来硬凑~~）

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210715231710259.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xSWTg5NzU3,size_16,color_FFFFFF,t_70#pic_center)

### 反向传播代码实现：

##### 全连接层反向传播：

```python
    # 全连接层的反向传播
    def backward(self, dout):
        '''
        input:
        dout: 损失函数相对于全连接层输出的梯度，形状为(B,M)，M是全连接层的输出神经元个数。
        在前向传播时全连接层的输入记录在了self.x中，故由此我们可以利用dout和self.x得到W的梯度
        output:
        dx:  (B, N) 关于输入层的梯度，便于进一步反向传播
        self.W和self.b的梯度分别存储在self.dW和self.db中
        self.dW: (N, M) 与self.W形状相同，self.W的梯度
        self.db: (M,)， self.b的梯度
        将x的梯度返回。
        '''
        # 以下所有即为矩阵的求导方法，我们也可以根据形状输入输出求解
        # 均依赖于公式 Y = X^T + W
        self.db = np.sum(dout, axis=0)    # 需要将得到的所有dout延y轴相加，因为取loss是就除以了batch_size
        self.dW = np.dot(self.x.T, dout)  # 在前向传播时全连接层的输入记录在了self.x中，这一项根据矩阵求导得到我们结果
        dx = np.dot(dout, self.W.T)       # 由矩阵求导得出结果
        return dx                       # 返回对输入层求导的结果，便于记录进一步反向传播
```

dout是损失函数相对于全连接层输出的梯度，形状为(B,M)，M是全连接层的输出神经元个数。由于在前向传播时全连接层的输入记录在了self.x中，故由此我们可以利用dout和self.x得到W的梯度。而我们三行关键的代码都是依赖于Y = $$X^T$$W + b公式得到的，另外加上一点矩阵求导的知识。值得注意的是db的求解，是dout沿着y轴叠加，因为之前已经取过平均值，这次不需再取。

##### 激活函数反向传播：

```python
    # sigmoid的反向传播
    def backward(self, dout):
        '''
        input: 
        dout：损失函数相对于sigmoid输出的梯度
        output:
        dz:相对于矩阵y得到的梯度
        '''
        dz = dout * self.z * (np.ones(self.z.shape) - self.z)
        return dz
```

实际上也就是我们得原梯度乘以y(1-y),这里利用矩阵的对应项相乘刚刚好。

```python
    def backward(self, dout):
        '''
        input: 
        dout：损失函数相对于relu输出的梯度
        output:
        dz:相对于矩阵y得到的梯度
        '''
        dout[self.mask] = 0
        dz = dout
        return dz
```

 relu反向传播更简单，只需要将小于零得元素的对应位置改为0即可，实际上就是利用我们之前self.mask里存的个元素是否小于零的真值表来直接赋值即可。

### Softmax回归+交叉熵

#### 前向传播

softmax实际属于分类，主要方便我们分完类后求误差损失函数.且有$$0<=\hat{y_i}<=1$$,并且和为1.这样就可以用描述概率的分布来描述这些量了，损失函数就可以使用交叉熵来描述了。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210715234635475.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xSWTg5NzU3,size_16,color_FFFFFF,t_70#pic_center)

##### softmax函数有关代码：

```python
# 激活函数softmax
def softmax(y):
    '''
    input:
    y:最终得到的预测输出结果矩阵
    output:
    将其使用softmax归一化返回处理后的矩阵（利于计算损失函数）
    '''
    y = y - np.max(y, axis=1, keepdims=True)     # 防止产生exp溢出的危险，所以每一行都减去最大值，且由加减值性质易得不会对值产生影响
    return np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)   # 返回softmax处理后矩阵，利于进一步计算损失函数
```

这里由于softmax函数的特有性质，加减一个常数没有影响，我们为了防止溢出，就提前减去最大值。



交叉熵函数可以很好用来描述概率分布差异的函数，当使用one-hot编码的时候，其结果可以很直接的化简为一项：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210715235153373.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xSWTg5NzU3,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210715235153361.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xSWTg5NzU3,size_16,color_FFFFFF,t_70#pic_center)

##### softmax与交叉熵前向传播代码

所以可以得到有关softmax与交叉熵代码：

```python
    # SoftMax + Cross Entropy的前向传播
    def forward(self, y, label):
        '''
        input:
        y : 预测输出结果矩阵，需要进一步softmax处理并利用其求出误差损失，形状为(B, 10)
        label: 真实标签矩阵，形状为 (B， 1)
        output:
        loss: 交叉熵损失
        '''
        z = softmax(y)      # 使用激活函数将输出矩阵归一化
        batch_size = z.shape[0]   # 得到batch_size
        loss = -np.sum(np.log(z[np.arange(batch_size), label])) / batch_size    # 求出平均损失误差值，使用交叉熵，利用one-hot特性得到每组输入的log值
#                 loss = -np.sum(np.log(z[np.arange(batch_size), t] + 1e-7)) / batch_size
        self.loss = loss  # 记录损失值
        self.z = z     
        self.label = label    # 存储记录
        return loss    # 返回误差损失
```

误差损失注意有一个负号，另外就是我们可以利用one-hot的特性来直接仅需加上特定位置的元素，其余位置均为0.

#### 反向传播

##### 反向传播推导

对softmax单独求导比较复杂一些，需要分类考虑：

![在这里插入图片描述](https://img-blog.csdnimg.cn/2021071600022349.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xSWTg5NzU3,size_16,color_FFFFFF,t_70#pic_center)

主要分为两种情况，这是对softmax单独求导，如果结合了交叉熵的话，就会简单很多：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210716113441260.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xSWTg5NzU3,size_16,color_FFFFFF,t_70#pic_center)

可以看到结合了交叉熵和onehot编码之后，我们的求导立刻变得简单多了，几乎不需要什么变化，只需要到时候将标签集对应位置的值减1即可，所以代码就好实现了。

##### 反向传播

由以上推导我们可以进一步得到我们的代码：

```python
    # SoftMax + Cross Entropy的反向传播
    def backward(self):
        '''
        output:
        交叉熵+softmax梯度
        '''
        batch_size = self.z.shape[0]  # 得到batch_size
        dz = np.copy(self.z)       # 深拷贝
        for label_, z_ in zip(self.label, dz):   # 由求梯度+onehot编码推出仅需在真实值所在位置减1即得梯度
            z_[label_] -= 1
        dz /= batch_size   # 取平均
        return dz   # 返回梯度
```

这里面我们仅需在对应位置减一即可，利用zip函数，对于每一条数据集的对应减一，之后求一个平均值就得到我们的梯度，并且返回即可。





### 最终神经网络类实现

#### 初始化

由于之前对于每一个激活函数、全连接层、softmax函数都实现了一个类，所以我们实现神经网络仅需调用相关类实现就可。

首先是初始化有关参数，我们有两个隐藏层，所以我们定义有关神经元的个数，以及超参数学习率这些：

```python
# 神经网络实现
class Network:
    # 初始化
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size=10, lr=0.1):
        '''
        Parameters:
        input_size, hidden_size1, hidden_size2, output_size:
        分别为输入层神经元个数、隐藏层神经元数、隐藏层神经元个数、输出层神经元个数数(手写数字识别默认为10), 学习率(默认为0.1)
        output:None
        '''
        W1 = np.random.randn(input_size, hidden_size1)  # 随机初始化权重
        W2 = np.random.randn(hidden_size1, hidden_size2)
        W3 = np.random.randn(hidden_size2, output_size)
        b1 = np.random.randn(hidden_size1)
        b2 = np.random.randn(hidden_size2)
        b3 = np.random.randn(output_size)
        
        
        self.lr = lr     # 学习率
        self.layer_1 = FullyConnected(W1, b1)
        self.sigmoid_1 = Sigmoid()
        self.layer_2 = FullyConnected(W2, b2)
        self.sigmoid_2 = Sigmoid()
        self.layer_last = FullyConnected(W3, b3)
        self.loss = SoftmaxWithLoss()
```

由于是手写数字，最终输出神经元个数为10， 而学习率我们默认为0.1.不行的话我们可以继续调。首先就是初始化权重矩阵，接着就是调用有关类，实现有关类的初始化。

#### 前向传播

```python
    # 神经网络前向传播
    def forward(self, x, label):
        '''
        input:
        x: 形状为(B,N)，输入的原始数据， B为批量Batch_size
        label:输入B个数据的分类类别，形状为(B, 1)
        output:
        最后输出的预测向量以及我们得到的误差
        '''
        y1 = self.layer_1.forward(x)  # 前向传播，一步步往后走
        z1 = self.sigmoid_1.forward(y1)
        y2 = self.layer_2.forward(z1)
        z2 = self.sigmoid_2.forward(y2)
        y3 = self.layer_last.forward(z2)
        loss = self.loss.forward(y3, label)
        
        return y3, loss
```

这便是遵循了前向传播的流程，一步一步往后走，具体流程如下:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210716115146420.png#pic_center)

返回值为预测输出与损失值。

#### 反向传播

```python
    # 神经网络反向传播
    def backward(self):
        '''
        input:None
        output:
        各项参数的梯度
        '''
        d = self.loss.backward()     # 反向传播，一步步往前走，和前向完全相反
        d = self.layer_last.backward(d)
        d = self.sigmoid_2.backward(d)
        d = self.layer_2.backward(d)
        d = self.sigmoid_1.backward(d)
        d = self.layer_1.backward(d)    # 至此，我们单次反向传播完成。
        
        return self.layer_1.dW, self.layer_1.db, self.layer_2.dW, self.layer_2.db, self.layer_last.dW, self.layer_last.db  # 将每层间权重的W 、偏移量b梯度返回
```

与前向传播完全相反，依次调用，我们各层的梯度都存储到具体的类之中，等到更新的时候调用就行。

#### 更新权重

```python
# 神经网络更新权重
def refresh(self):
    lr = self.lr   # 得到学习率
    self.layer_1.W -= lr * self.layer_1.dW   # 更新每一个参数权重
    self.layer_1.b -= lr * self.layer_1.db
    self.layer_2.W -= lr * self.layer_2.dW
    self.layer_2.b -= lr * self.layer_2.db
    self.layer_last.W -= lr * self.layer_last.dW
    self.layer_last.b -= lr * self.layer_last.db
```

利用梯度更新权重。

#### 训练模型并测试准确率

```python
    # 预测模型
    def predict(self, test_images, test_labels):
        '''
        '''
        pred, loss = self.forward(test_images, test_labels)  # 预测值和损失
        pred = np.argmax(pred, axis=1)   # 求出预测标签
        return pred, loss, right_rate(pred, test_labels)   # 返回预测值向量和损失误差以及正确率。
        # 训练模型并判断正确率
        
        
    def fit_pred(self, train_images, train_labels, test_images, test_labels, Epochs=5, batch_size=100):
        '''
        input：
        train_images, train_labels:训练集
        test_images, test_labels:测试集
        Epochs:扫多少遍训练集
        batch_size：批量大小
        output:None
        '''
        samples_num = train_images.shape[0]   # 得到训练集数量
        for epoch in range(1, Epochs + 1):   # 在训练集里面跑5次
            i = 0
            while i < samples_num:
                self.forward(train_images[i:i+batch_size], train_labels[i:i+batch_size])  # 每次训练batch_size个样本
                self.backward()         # 反向传播
                self.refresh()          # 更新参数
#                 print("Train Epoch: {}\t batch_size_index:{} Loss:{:.6f}".format(epoch, i+1, self.loss.loss))
                i += batch_size
                
#             self.lr = (0.95 ** epoch) * self.lr     # 更新学习率，防止其因为学习率过大而导致无法有效下降。
            print("Train Epoch: {}\t Loss:{:.6f}".format(epoch, self.loss.loss))
            pred, pred_loss, right_rate = self.predict(test_images, test_labels)    # 计算测试集精度
            print("Test -- Average loss:{:.4f}, Accuracy:{:.3f}\n".format(pred_loss, right_rate))
```
