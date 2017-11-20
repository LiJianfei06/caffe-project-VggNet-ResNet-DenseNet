

# -*- coding: UTF-8 -*-
import sys
                                       

sys.path.append("/home/ljf/caffe-master/python")
sys.path.append("/home/ljf/caffe-master/python/caffe")
import caffe     
from caffe import layers as L,params as P,to_proto
import tools






root_str="/home/ljf/caffe-master/examples/ljftest_cifar10_DenseNet/"



if __name__ == '__main__':
    caffe.set_device(0)
    caffe.set_mode_gpu()#设置迭代采用 GPU 加速模式
#选择 caffe 模型，这里选择第 65000 次迭代的数据
    snapshot_model_dir = root_str +'model_save/caffe_ljftest_train_iter_200000.caffemodel'
#我们只关注测试阶段的结果，因此只写入 test.prototxt
    test_prototxt_dir = root_str+"test.prototxt" 
    net = caffe.Net(str(test_prototxt_dir), str(snapshot_model_dir), caffe.TEST)
    sum = 0
#测试 79 次，取平均值
    for _ in range(100):
        net.forward()
        sum += net.blobs['Accuracy1'].data
        print net.blobs['Accuracy1'].data
    sum /= 100 
    print "sum:",sum
