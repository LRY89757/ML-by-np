# Decision Tree

---

逯润雨    计卓2001

事先声明：***水印是本人博客，为了markdown能正常显示图片，本人先将图片放到博客上了。***

## 数据处理部分

### 读取数据

读取数据是参考的第一题题目提供的读取Kmeans数据集代码，我稍微改了一下然后写到了这里。

```python
# 读取数据
class READ(object):
    '''
    sonar数据集
    '''
    def __init__(self,root,path):
        '''
        方法说明:
            初始化类
        参数说明:
            root: 文件夹根目录
            path: sonar数据集文件名 'sonar.csv'
        '''
        self.root = root
        self.path = path
        self.feature, self.label = self._get_data()

    def _get_data(self):
        #打开数据集
        with open(os.path.join(self.root,self.path),'r') as f:
            data = f.readlines()[:]
        feature = []
        label = []
        #去除掉逗号
        for i in range(len(data)):
            feature.append(data[i].strip().split(',')[:-1])
            label.append(data[i].strip().split(',')[-1])
        #转化为numpy数组格式
        feature = np.array(feature)
        label = np.array(label)
        
        return feature, label

root = "./"
path = "sonar.csv"
dataset = READ(root, path)
feature = dataset.feature
label = dataset.label
```



### 处理连续型数据变为离散型

首先是将我们的标签“M”、“R”数字化：

```python
# 处理数据，将数据标签数字化
trans_label = {'M':0, 'R':1}
label = np.array([trans_label[value] for value in label])   # 数字化label：  'M'：0， ’R‘：1
```

接着是我们数据集进行数字化，基本思路是这样的，将每一个属性取最大值最小值，然后每一个属性的最大最小值中间分为若干个大小相等的区间，然后对属性依次标上序号0、1、2……经过多次实验之后，发现平均分为4份最终得到的正确率最高，所以最后确定为4.

```python
# # 将连续型数据变为离散型
feature_range = [(min(sing), max(sing)) for sing in feature.T]   # 每一个属性的最大值最小值的列表
# print(feature_range)
range_divide = [np.arange(low, high+0.5, (high - low) / 4) for low, high in feature_range] # 分为10等分，每个属性都有10类
# print(range_divide.shape)
# 将每一类连续型的变为离散型
def trans_feature(sing_feature, range_divide, index):      
    '''
    input:
    sing_feature:第index个属性的类别矩阵
    range_divide:划分标准矩阵
    output:划分好的离散型
    '''
    output = []
    for sing in sing_feature:
        for i in range(len(range_divide[index])):
            if range_divide[index][i] > sing:
                output.append(i - 1)
                break
    return np.array(output)
# len(trans_feature(feature.T[0], range_divide, 0))
feature = np.array([trans_feature(feature.T[index], range_divide, index) for index in range(len(feature.T))]).T
# print(type(feature[0][0]))
# feature = np.array([np.array([feature[i][index] for i in range[60]]) for index in range(208)])
print(type(feature))
print(feature.shape)
```

### 随机抽取生成训练集、测试集、验证集

这里采取np.random.choice作为抽取的主要函数。

```python
# 随机抽取数据作为训练集和测试集以及验证集
train_val = np.random.choice(range(208), 150, replace=False)        # 训练集和验证集索引
val = np.random.choice(range(70), 50, replace=False)
train = np.random.choice([i for i in range(150) if i not in val], 90, replace=False)
test = np.random.choice([i for i in range(208) if i not in train_val], 58, replace=False) # 测试集索引
# print(feature.shape)
train_feature = feature[train]       
train_label = label[train]       # 训练集
test_feature = feature[test]
test_label = label[test]         # 测试集
val_feature = feature[val]        
val_label = label[val]          # 验证集

# 观察数据规模
print(train_feature.shape)
print(feature.shape)
print(label.shape)
```

### 正确率函数

计算预测值的正确率。

```python
# 计算预测数据正确率
def right_rate(pred_label, label):
    '''
    input:
    pred_label:预测的结果数组
    label:实际的类别数组
    output:
    rate:正确率
    '''
    diff = pred_label - label
    return diff.tolist().count(0) / len(diff)
```



## 决策树代码（非剪枝）

代码思路按照测试题目所提供的参考资料的伪代码：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210719145331655.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xSWTg5NzU3,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210719145331897.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xSWTg5NzU3,size_16,color_FFFFFF,t_70#pic_center)



选取最优划分属性按照信息增益，也即ID3算法：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210719145821974.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xSWTg5NzU3,size_16,color_FFFFFF,t_70#pic_center)



下面是代码具体分析：

### 类的初始化：

定义模型，使用字典来存储

```python
class DecisionTree(object):
    def __init__(self):
        #决策树模型
        self.tree = {}
```

### 计算信息熵

首先定义了一个计算pklogpk的函数，其次便是计算出每一个种类的概率，最后就是利用map函数对于整体进行运算算出每一个pklogpk，利用sum函数相加。返回结果。

```python
    # 计算信息熵
    def cal_Ent(self, feature, label):
        '''
        input:feature 特征数据集
              label 标签集
        output:信息熵
        '''
        # 计算信息熵的pklogpk函数
        def operate(pk):
            '''
            input:概率pk
            output:pklogpk
            '''
            if pk == 0:           # 定义0log0=0
                return 0
            else:                 # 计算信息熵
                return 0 - pk * math.log(pk, 2)
        
        varies = list(set(label))       # 含有的种类的列表（不重复）
        p_labels = [label.tolist().count(vary) / len(label) for vary in varies]   # 每一个种类的概率矩阵

        return sum(list(map(operate, p_labels)))      # 计算出信息熵
```



### 计算信息增益

首先是提取出我们要的特征，这时候我们需要将feature转置，然后利用index提取出这个属性的所有值，接着计算出该属性总样本的信息熵，而后找出该属性所有种类，接着划分出各个种类的索引，索引的向量矩阵放到一个列表中，而后就是对于每一个种类都算出其信息熵，而后利用sum加起来，最后返回Ent_old - Ent_new也就是信息增益。

```python
    # 计算信息增益
    def cal_InfoGain(self, feature, label, index):
        '''
        input：
         feature:特征数据集
         label:标签集
         index:feature部分特征列的索引。该索引指的是feature中第几个特征，如index:0表示使用第一个特征来计算信息增益。
        output:信息增益
        '''
        
        feature_ = np.array(feature).T[index]       # 把待查找的特征提取出来
        Ent_old = self.cal_Ent(feature_, label)   # 这是总样本的信息熵

        varies = list(set(feature_.tolist()))   # 讲样本该属性的种类列表找出来
        feature_sorted = [np.where(feature_ == value)[0] for value in varies]  # 把分类后各小样本的索引列表在放到一个大列表中   

        Ent_new = sum([self.cal_Ent(feature_[value], label[value]) * len(label[value]) / len(feature_) for value in feature_sorted])     # 这是分类后各小样本的信息熵之和               
        
        return Ent_old - Ent_new      # 返回信息增益
```

### 获取信息增益最高的特征

对每一个属性都计算信息增益而后计算出算出信息增益最大值的索引。利用np.argmax函数得到最大值的索引。

```python
 # 获得信息增益最高的特征
    def getBestFeature(self, feature, label):
        '''
        input:feature 特征数据集
              label 标签集
        output:信息熵
        '''
        return np.argmax([self.cal_InfoGain(feature, label, index) for index in range(len(feature[0]))])   # 返回最大信息增益的特征的索引
```



### 创建决策树

这是一个递归的函数，首先如果各个样本的种类相同，那么这个就是叶子节点，返回该节点label值就行。而后如果我们的样本中只有一个特征了，或是样本中的特征的取值都一样了就选择出现次数最多的label作为叶子节点的值（利用max(a, key=a.count)函数）。以上是终止递归的条件，而后便是我们的正常构造，首先根据信息增益选取最优的属性的索引，接着建立一个关于该属性的结点，这里的树结构是利用字典存储的。接着将该属性的特征向量提取出来，得到该属性所含种类的列表，接下来按照种类创建分支结点，按照种类分类，创建完毕后循环各个结点开始递归循环创建各个子树。最后返回根结点本身。

```python
 # 创建决策树
    def createTree(self, feature, label):
        '''
        input:feature 特征数据集
              label 标签集
        output:信息熵
        '''
        
        if len(set(label)) == 1:           # 样本里都是同一个label没必要继续分叉了
            return label[0]              # 直接作为叶子节点返回
        
        if len(feature[0]) == 1 or len(np.unique(feature, axis=0)) == 1:   # 样本中只有一个特征或者所有样本的特征都一样的话就看哪个label的数量比较多
            return max(label, key=label.tolist().count)   # 返回出现次数最多的label
        
        best_feature = self.getBestFeature(feature, label)   # 根据信息增益得到特征的索引
        tree = {best_feature: {}}      # 建立结点

        feature_ = feature[:, best_feature]   # 将信息增益最大的特征的特征向量提取出来
        varies = list(set(feature_))          # 含有所有的种类（不重复）的列表
        sub_features = {vary: np.where(feature_ == vary)[0] for vary in varies}   # 把feature按照该特征分类，key是种类，value是索引的ndarray
        for vary in varies:
            tree[best_feature][vary] = self.createTree(feature[sub_features[vary]], label[sub_features[vary]])  # 递归求解构造新树 
        
        return tree
```



### 训练模型

就是创建决策树的过程。

```python
    # 训练模型
    def fit(self, feature, label):
        '''
        input: feature: 训练集数据
         label:训练集标签
        output: None
        '''
        self.tree = self.createTree(feature, label)
```



### 预测函数

就是一个相当于数据结构二叉树查找的过程，只不过这里的查找是n叉树，仅需我们先定义一个有关针对每一条数据进行查找的递归函数，而后使用列表推导式进行每一条数据的查找返回即可。不过这个查找函数需要我们注意如果找不到该属性对应的索引，可以根据就近原则查找离他最近的那个索引。

```python
    # 预测
    def predict(self, feature):
        '''
        input: feature:测试集数据
        output:预测结果，如np.array([0, 1, 2, 2, 1, 0])
        '''
        def judge_sing_feature(sing_feature, tree):
            '''
            input:sing_feature:单条数据
            output:类别sing_label
            '''
            tree = tree
            for k in tree.keys():
                try:                      # 如果tree[k]这个字典中有sing_feature[k]这个关键字
                    if isinstance(tree[k][sing_feature[k]], dict) == False:     # 不是字典类型就是值，则返回种类
                        return tree[k][sing_feature[k]]          # 返回种类
                    else:
                        tree = judge_sing_feature(sing_feature, tree[k][sing_feature[k]])          # 是字典的话继续递归
                except:                # 否则找出字典tree[k]的键中离sing_feature[k]最近的一个key
                        k_ = list(tree[k].keys())[np.argmin(np.array([(key - sing_feature[k]) ** 2 for key in list(tree[k].keys())]))]   # 字典tree[k]的键中离sing_feature[k]最近的一个key
                        if isinstance(tree[k][k_], dict) == False:      # 不是字典类型就是值，则返回种类
                            return tree[k][k_]          # 返回种类
                        else:
                            tree = judge_sing_feature(sing_feature, tree[k][k_])  # 是字典的话继续递归
            return tree
        
        return np.array([judge_sing_feature(sing_feature, self.tree) for sing_feature in feature])  # 返回预测结果
```



### 训练、测试结果

```python
A = DecisionTree()
A.fit(train_feature, train_label)  # 训练决策树
# print(A.tree[0])
# print(A.tree)
# for k in A.tree.keys():
#     print(A.tree[k])

pred_label = A.predict(test_feature)  # 预测
# print(test_label)
# print(pred_label)
# print(test_label)
# print(test_label)
# print(train_label)
# right_rate(pred_label, test_label)
# print(A.tree)
# print(pred_label.shape, label.shape)
print(right_rate(pred_label, test_label))  # 输出正确率
# right_rate(pred_label, test_label)
# print(pred_label)
# print(label)
```

预测结果为：

```python
0.7586206896551724
```

正确率为0.758左右





## 决策树代码（后剪枝）

首先吐槽一下：剪枝太难了😭……我花在剪枝里的时间比在神经网络还高……由于与之前代码有许多重叠的，这里只介绍后剪枝的代码。

剪枝思路还比较简单，“就是”防止过拟合，将一些可能没用的叶子节点合并在一起，如果合并在一起预测率（用验证集验证）还高一点，那就合并，反之就不合并。但是具体实现起来却不是很容易……

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210719153807753.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xSWTg5NzU3,size_16,color_FFFFFF,t_70#pic_center)



### 后剪枝总代码

```python
     # 计算验证集准确率
    def calc_acc_val(self, tree, val_feature, val_label):
        '''
        input:
        tree:决策树
        val_feature:验证集的数据
        val_label：验证集的标签集
        output：决策树正确率
        '''
        def classify(tree, sing_feature):
            '''
            input: 
            tree:决策树
            sing_feature:单条数据
            output：正确率            
            '''
            if not isinstance(tree, dict):       # 如果tree直接就是值而不是字典，直接返回分类值
                return tree
            index, value = list(tree.items())[0]     # 得到tree的键和值
            f_value = sing_feature[index]             # 得到该属性的feature的值，以便接下来归类
            if isinstance(value, dict):       # 如果tree内含的值对应的是字典
                try:                  # 如果tree[index]含有f_value的键
                    classLabel = classify(tree[index][f_value], sing_feature)  # 递归查找
                    return classLabel
                except:              # 如果tree[index]不含有f_value的键，找到与其最接近的属性
                    f_value_ = list(tree[index].keys())[np.argmin(np.array([(key - f_value) ** 2 for key in list(tree[index].keys())]))]   # 字典tree[index]的键中离sing_feature[index]最近的一个key
                    classLabel = classify(tree[index][f_value_], feature)
                    return classLabel
            else:                         
                return value           # 如果tree内含的值对应的不是字典，直接返回值
            
        return right_rate([classify(tree, sing_feature) for sing_feature in val_feature], val_label)   # 返回正确率
    
    
    
    
    
    # 后剪枝
    def post_cut(self, val_feature, val_label):
        '''
        input:
        val_feature:验证集的数据
        val_label：验证集的标签集
        output：None
        '''
        
        # 深度优先搜索
        def dfs(tree, path, all_path):       # 深度优先搜索
            '''
            input: 
            tree:决策树
            path:记录深度遍历的路径
            all_path:记录每一条路径
            output：None
            '''
            for k in tree.keys():          # 对于所有键（实际上是所有的子树）搜索
                if isinstance(tree[k], dict):   # 如果是字典，也就是不是叶子节点
                    path.append(k)         # 将该路径加进来
                    dfs(tree[k], path, all_path)   # 递归调用
                    if len(path) > 0:      # 如果path不为空
                        path.pop()         # 如果遍历之后，那么就将其弹出，这样回退到最开始的时候path就为空，回退到上一级时候可以继续记录遍历同级的其他的子树了
                else:
                    all_path.append(path[:])   # 到最深处无法回溯后，将这一条路径放到all_path,得到了一个非叶子节点的路径
        
        # 拿到非叶子节点的数量
        def get_non_leaf_node_count(tree):
            '''
            output:tree：生成的决策树
            input: 决策树中通往所有非叶子节点的路径
            '''
            non_leaf_node_path = []
            dfs(tree, [], non_leaf_node_path)    # 调用函数得到所有通往非叶子节点的路径。
            unique_non_leaf_node = []     # 得到通往每一个叶子结点的路径，不重复
            for path in non_leaf_node_path:
                if path in unique_non_leaf_node:   # 如果已经有了
                    continue                    # 没有任何操作，直接跳过
                unique_non_leaf_node.append(path)   # 如果没有就加上
            
#             print(non_leaf_node_path)
#             print(unique_non_leaf_node)
            
            return unique_non_leaf_node   # 返回路径长度
        
        
        # 拿到树中深度最深的从根节点到非叶子节点的路径
        def get_the_most_deep_path(tree):
            '''
            input:tree:决策树
            output:将最深路径输出
            '''
            non_leaf_node_path = []
            dfs(tree, [], non_leaf_node_path)   # 深搜得到所有路径
            return non_leaf_node_path[np.argmax(np.array([len(path) for path in non_leaf_node_path]))]  # 返回长度最长的路径的索引，然后访问返回它

        # 剪枝
        def set_vote_label(tree, path, max_label):
            '''
            input：
            tree:当前决策树树
            path:记录深度遍历的路径
            max_label:原非叶子含有数目种类最多的标签
            '''
            for i in range(len(path)-1):      
                tree = tree[path[i]]
            tree[path[-1]] = max_label     # 非叶子节点赋值
        
#         path_visited = []    # 记录已经访问过的路径
        all_path_ = get_non_leaf_node_count(self.tree)  # 记录通往所有非叶子结点的路径

        acc_before_cut = self.calc_acc_val(self.tree, val_feature, val_label)   # 计算目前的正确率
        # 遍历所有非叶子节点
        for i in range(len(all_path_)):
#             path = get_the_most_deep_path(self.tree)    # 得到最深的路径
            path = all_path_[len(all_path_) - i - 1]
#             path_visited.append(path)
#             print(path)
            
            tree = deepcopy(self.tree)    # 将树完全复制一遍，另外开了一个存储空间，防止改变原树数据
            step = deepcopy(tree)         # 同理
            
            for k in path:
                step = step[k]      # 跟着路径走
            
            flag = False              # 判断是否该path是的子树全是叶子结点
            for value in step.values():
                if isinstance(value, dict):
                    flag = True
            if flag:         # 如果不是那么就返回
                continue
            
            max_label = max(list(step.values()), key=list(step.values()).count)   # 叶子节点中票数最多的标签
#             print(max_label)
            
            set_vote_label(tree, path, max_label)           # 在备份的树上剪枝
            acc_after_cut = self.calc_acc_val(tree, val_feature, val_label)   # 计算剪枝之后的正确率
#             print(self.tree)
#             print(tree)
#             print('hello world')
#             print(acc_after_cut, acc_before_cut)
            
            if acc_after_cut > acc_before_cut:            # 验证集准确率高于原来的就剪枝
                set_vote_label(self.tree, path, max_label)   # 剪枝
                acc_before_cut = acc_after_cut            # 剪完后正确率更新
#                 print('hello world')
```

以下为详细分析：

---

### 计算验证集准确率

这个实际上和我们之前非剪枝的predict函数是非常像的，~~不能说完全相同，可以说是一模一样~~，都是采用了一个递归查找单条数据的结果，然后使用列表推导式得到每一个预测结果，最后和真实值相比计算正确率。

```python
    # 计算验证集准确率
    def calc_acc_val(self, tree, val_feature, val_label):
        '''
        input:
        tree:决策树
        val_feature:验证集的数据
        val_label：验证集的标签集
        output：决策树正确率
        '''
        def classify(tree, sing_feature):
            '''
            input: 
            tree:决策树
            sing_feature:单条数据
            output：正确率            
            '''
            if not isinstance(tree, dict):       # 如果tree直接就是值而不是字典，直接返回分类值
                return tree
            index, value = list(tree.items())[0]     # 得到tree的键和值
            f_value = sing_feature[index]             # 得到该属性的feature的值，以便接下来归类
            if isinstance(value, dict):       # 如果tree内含的值对应的是字典
                try:                  # 如果tree[index]含有f_value的键
                    classLabel = classify(tree[index][f_value], sing_feature)  # 递归查找
                    return classLabel
                except:              # 如果tree[index]不含有f_value的键，找到与其最接近的属性
                    f_value_ = list(tree[index].keys())[np.argmin(np.array([(key - f_value) ** 2 for key in list(tree[index].keys())]))]   # 字典tree[index]的键中离sing_feature[index]最近的一个key
                    classLabel = classify(tree[index][f_value_], feature)
                    return classLabel
            else:                         
                return value           # 如果tree内含的值对应的不是字典，直接返回值
            
        return right_rate([classify(tree, sing_feature) for sing_feature in val_feature], val_label)   # 返回正确率
```



### 深度优先搜索

这里深搜的目的就是为了找到所有非叶子节点的路径。具体解释见每一行代码，我都加了注释。~~（实际上前面的代码我也几乎都加了注释）~~

```python
 # 深度优先搜索
        def dfs(tree, path, all_path):       # 深度优先搜索
            '''
            input: 
            tree:决策树
            path:记录深度遍历的路径
            all_path:记录每一条路径
            output：None
            '''
            for k in tree.keys():          # 对于所有键（实际上是所有的子树）搜索
                if isinstance(tree[k], dict):   # 如果是字典，也就是不是叶子节点
                    path.append(k)         # 将该路径加进来
                    dfs(tree[k], path, all_path)   # 递归调用
                    if len(path) > 0:      # 如果path不为空
                        path.pop()         # 如果遍历之后，那么就将其弹出，这样回退到最开始的时候path就为空，回退到上一级时候可以继续记录遍历同级的其他的子树了
                else:
                    all_path.append(path[:])   # 到最深处无法回溯后，将这一条路径放到all_path,得到了一个非叶子节点的路径
```



### 拿到通往所有非叶子节点的路径

这里面通过dfs已经拿到了所有的路径，但是我们需要对其进行去重操作，因为这之中有许多重复的路径，但是去重的话似乎不能直接用set函数，我只能一个一个查找去重。最后返回所有路径。

```python
        # 拿到通往所有非叶子节点的路径
        def get_non_leaf_node_count(tree):
            '''
            output:tree：生成的决策树
            input: 决策树中通往所有非叶子节点的路径
            '''
            non_leaf_node_path = []
            dfs(tree, [], non_leaf_node_path)    # 调用函数得到所有通往非叶子节点的路径。
            unique_non_leaf_node = []     # 得到通往每一个叶子结点的路径，不重复
            for path in non_leaf_node_path:
                if path in unique_non_leaf_node:   # 如果已经有了
                    continue                    # 没有任何操作，直接跳过
                unique_non_leaf_node.append(path)   # 如果没有就加上
            
#             print(non_leaf_node_path)
#             print(unique_non_leaf_node)
            
            return unique_non_leaf_node   # 返回所有路径
```



### 剪枝操作

先走到叶子结点的父节点，而后将父节点的孩子赋值为孩子节点中label最多的种类。

```python
  		# 剪枝
        def tree_cut(tree, path, max_label):
            '''
            input：
            tree:当前决策树树
            path:记录深度遍历的路径
            max_label:原非叶子含有数目种类最多的标签
            '''
            for i in range(len(path)-1):      
                tree = tree[path[i]]
            tree[path[-1]] = max_label     # 非叶子节点赋值
```

### 循环过程

首先得到通往所有非叶子节点路径，接着计算目前的正确率，而后依次遍历路径，先另开一个存储空间复制一遍树，而后判断是否沿着该路径走得到的子树是否都是叶子节点，如果不是那么就重新遍历，如果是那么就先计算出叶子节点中最多的label，接着丢进剪枝函数剪枝，剪完之后判断正确率是否比之前的大，如果大的话就将这个树赋值给self.tree，不大的话就继续遍历其他路径。

```python
#         path_visited = []    # 记录已经访问过的路径
        all_path_ = get_non_leaf_node_count(self.tree)  # 记录通往所有非叶子结点的路径

        acc_before_cut = self.calc_acc_val(self.tree, val_feature, val_label)   # 计算目前的正确率
        # 遍历所有非叶子节点
        for i in range(len(all_path_)):
            path = all_path_[len(all_path_) - i - 1]
#             path_visited.append(path)
#             print(path)
            
            tree = deepcopy(self.tree)    # 将树完全复制一遍，另外开了一个存储空间，防止改变原树数据
            step = deepcopy(tree)         # 同理
            
            for k in path:
                step = step[k]      # 跟着路径走
            
            flag = False              # 判断是否该path是的子树全是叶子结点
            for value in step.values():
                if isinstance(value, dict):
                    flag = True
            if flag:         # 如果不是那么就返回
                continue
            
            max_label = max(list(step.values()), key=list(step.values()).count)   # 叶子节点中票数最多的标签
#             print(max_label)
            
            tree_cut(tree, path, max_label)           # 在备份的树上剪枝
            acc_after_cut = self.calc_acc_val(tree, val_feature, val_label)   # 计算剪枝之后的正确率
#             print(self.tree)
#             print(tree)
#             print('hello world')
#             print(acc_after_cut, acc_before_cut)
            
            if acc_after_cut > acc_before_cut:            # 验证集准确率高于原来的就剪枝
                tree_cut(self.tree, path, max_label)   # 剪枝
                acc_before_cut = acc_after_cut            # 剪完后正确率更新
#                 print('hello world')
```





### 训练、测试结果

```python
A = DecisionTree()
A.fit(train_feature=train_feature, train_label=train_label,val_feature=val_feature, val_label=val_label)
# print(A.tree[0])
# print(A.tree)
# tree = A.tree
# for k, v in tree:
#     print(A.tree.keys())
#     print(A.tree.values())
#     tree = tree

# for k in A.tree.keys():
#     print(A.tree[k])

pred_label = A.predict(test_feature)
print(right_rate(pred_label, test_label))
```

结果为：

```python
0.6896551724137931
```

剪枝后正确率反而下降了，个人认为应该是训练集偏小了，不过我们不是说过，

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210719161020284.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xSWTg5NzU3,size_16,color_FFFFFF,t_70#pic_center)



# ~~结果不是最重要的！！~~





