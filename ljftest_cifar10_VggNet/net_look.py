# -*- coding: UTF-8 -*-
import sys
                                       

sys.path.append("/home/ljf/caffe-master/python")
sys.path.append("/home/ljf/caffe-master/python/caffe")
import caffe     
from caffe import layers as L,params as P,to_proto
import tools






root_str="/home/ljf/caffe-master/examples/ljftest_cifar10_VggNet/"



if __name__ == '__main__':
    caffe.set_device(0) #设置使用的 GPU 编号
    caffe.set_mode_gpu()#设置迭代采用 GPU 加速模式
    snapshot_model_dir = root_str +'model_save/caffe_ljftest_train_iter_190000.caffemodel'
    test_prototxt_dir = "/home/ljf/caffe-master/examples/ljftest_cifar10_VggNet/train.prototxt" 
    net = caffe.Net(str(test_prototxt_dir), str(snapshot_model_dir), caffe.TEST)   
    for name in net.blobs.keys(): #blobs 里面存储的都是每层的数据
        print name, net.blobs[name].data.shape
   
