'''
使用说明：

    本文件仅用于帮助不熟悉python语法的同学快速入门，
    主要用于读取MNIST数据集，
    可以直接使用 也可自行实现。

    
    使用时，可从外部调用相应函数，如:
        ## from dataset import MNIST
        ##
        ## dataset = MNIST(image_file, lable_file)
        ## dataset.normalize()
        ##
        ## img = dataset.img
        ## label = dataset.label
'''


import struct
import os
import numpy as np

class MNIST(object):
    '''
    MNIST数据集类
    '''
    def __init__(self,root, image_file, lable_file):
        '''
        方法说明:
            初始化类
        参数说明:
            root: 文件夹根目录
            image_file: mnist图像文件 'train-images.idx3-ubyte' 'test-images.idx3-ubyte'
            label_file: mnist标签文件 'train-labels.idx1-ubyte' 'test-labels.idx1-ubyte'
        '''
        self.img_file = os.path.join(root, image_file)
        self.label_file = os.path.join(root, lable_file)
        
        self.img = self._get_img()
        self.label = self._get_label()

    #读取图片
    def _get_img(self):

        with open(self.img_file,'rb') as fi:
            ImgFile = fi.read()
            head = struct.unpack_from('>IIII', ImgFile, 0)
            #定位数据开始位置
            offset = struct.calcsize('>IIII')
            ImgNum = head[1]
            width = head[2]
            height = head[3]
            #每张图片包含的像素点
            pixel = height*width
            bits = ImgNum * width * height
            bitsString = '>' + str(bits) + 'B'
            #读取文件信息
            images = struct.unpack_from(bitsString, ImgFile, offset)
            #转化为n*726矩阵
            images = np.reshape(images,[ImgNum,pixel])
        
        return images

    #读取标签
    def _get_label(self):

        with open(self.label_file,'rb') as fl:
            LableFile = fl.read()
            head = struct.unpack_from('>II', LableFile, 0)
            labelNum = head[1]
            #定位标签开始位置
            offset = struct.calcsize('>II')
            numString = '>' + str(labelNum) + "B"
            labels = struct.unpack_from(numString, LableFile, offset)
            #转化为1*n矩阵
            labels = np.reshape(labels, [labelNum])

        return labels

    #数据标准化
    def normalize(self):
        
        min = np.min(self.img, axis=1).reshape(-1,1)
        max = np.max(self.img, axis=1).reshape(-1,1)
        self.img = (self.img - min)/(max - min)

    #数据归一化
    def standardlize(self):
        
        mean = np.mean(self.img, axis=1).reshape(-1,1)
        var = np.var(self.img, axis=1).reshape(-1,1)
        self.img = (self.img-mean)/np.math.sqrt(var)