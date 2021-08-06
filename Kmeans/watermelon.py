'''
使用说明：

    本文件仅用于帮助不熟悉python语法的同学快速入门，
    主要用于读取watermelon数据集，
    可以直接使用 也可自行实现。

    
    使用时，可从外部调用相应函数，如:
        ## from watermelon import WATERMELON
        ##
        ## dataset = WATERMELON(root,path)
        ## data = dataset.data
'''

import os
import numpy as np

class WATERMELON(object):
    '''
    西瓜数据集
    '''
    def __init__(self,root,path):
        '''
        方法说明:
            初始化类
        参数说明:
            root: 文件夹根目录
            path: 西瓜数据集文件名 'watermelon.csv'
        '''
        self.root = root
        self.path = path
        self.data = self._get_data()

    def _get_data(self):
        #打开数据集
        with open(os.path.join(self.root,self.path),'r') as f:
            data = f.readlines()[1:]
        #去除掉西瓜编号，逗号
        for i in range(len(data)):
            data[i] = data[i].strip().split(',')[1:]
        #转化为numpy数组格式
        data = np.array(data,dtype=float)
        
        return data