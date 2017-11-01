# -*- coding: UTF-8 -*-
from numpy import *
import operator
import os
from os import listdir
import sys
sys.path.append("/home/ljf/caffe-master/python")
sys.path.append("/home/ljf/caffe-master/python/caffe")
import caffe
                                                    
import numpy as np

root_str="/home/ljf/caffe-master/examples/ljftest_alphabet_ResNet/"
str_place="/home/ljf/caffe-master/examples/ljftest_alphabet_ResNet/test/image/"


def test(my_project_root, deploy_proto):
    caffe_model = my_project_root + 'snapshot_ResNet_iter_150000.caffemodel'        #caffe_model文件的位置
    for dirpath, dirnames, filenames in os.walk(str_place):
        print "Directory:%s"%dirpath
        labels_filename = my_project_root + 'test/labels.txt'            #类别名称文件,将数字标签转换回类别名称
        net = caffe.Net(deploy_proto, caffe_model, caffe.TEST)                #加载model和deploy
        #图片预处理设置
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  #设定图片的shape格式(1,3,28,28)
        transformer.set_transpose('data', (2,0,1))                            #改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)
        #transformer.set_mean('data', np.load(root_str+'imagenet_mean.npy').mean(1).mean(1)) #减去均值操作
        transformer.set_raw_scale('data', 255.0)                                # 缩放到【0，255】之间
        transformer.set_channel_swap('data', (2,1,0))                       #交换通道，将图片由RGB变为BGR   
        
        ture_mun=0.0
        false_mun=0.0
        for filename in filenames:
            #img = my_project_root + 'test/1/1_20.png'                      #找一张待测图片
            img = str_place + filename                                      #找一张待测图片
            #print img
            im = caffe.io.load_image(img,color=True)                                      #加载图片
            net.blobs['data'].data[...] = transformer.preprocess('data',im)     #执行上面设置的图片预处理操作，并将图片载入到blob中
            out = net.forward()                                                    #执行测试
            
            labels = np.loadtxt(labels_filename, str, delimiter='\t')           #读取类别名称文件
           
            prob = net.blobs['Softmax1'].data[0].flatten()                             #取出最后一层（Softmax）属于某个类别的概率值
            #print "prob:",prob
            order = prob.argsort()[-1]                                          #将概率值排序，取出最大值所在的序号
            
            #print ord(filename[0]),ord(labels[order].split(' ')[1])
            if ((abs(ord(filename[0])-ord(labels[order].split(' ')[1]))==32 and ord(labels[order].split(' ')[1])>48) or (ord(filename[0])-ord(labels[order].split(' ')[1])==0)):
                hint="True" 
                ture_mun+=1
            else:
                hint="False"
                false_mun+=1            
            
            
            
            print filename
            print("正确: {}  预测: {} check:{}".format(filename[0], labels[order].split(' ')[1],hint))
        print("accuracy:%.4f"%(ture_mun/(ture_mun+false_mun)))
if __name__ == '__main__':
    my_project_root = root_str    #my-caffe-project目录
    deploy_proto = my_project_root + "deploy.prototxt"            #保存deploy.prototxt文件的位置
    test(my_project_root, deploy_proto)
