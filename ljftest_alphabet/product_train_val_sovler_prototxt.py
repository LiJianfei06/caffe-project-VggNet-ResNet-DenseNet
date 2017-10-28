# -*- coding: UTF-8 -*-
import sys
sys.path.append("/home/ljf/caffe-master/python")
sys.path.append("/home/ljf/caffe-master/python/caffe")
import caffe     
from caffe import layers as L,params as P,to_proto
                                                #导入caffe包

root_str="/home/ljf/caffe-master/examples/ljftest_alphabet/"

def create_net(train_lmdb,val_lmdb, mean_file, train_batch_size=64,val_batch_size=50,include_type="train_val"):
    #网络规范
    net = caffe.NetSpec()
    if not include_type=="deploy":
        if not include_type=="deploy_fc":
            #第一层Data层
            net.data, net.label = caffe.layers.Data(name="face",source=train_lmdb, backend=caffe.params.Data.LMDB, batch_size=train_batch_size, ntop=2,   
                                                image_data_param=dict(shuffle=True),
                                                include={'phase':caffe.TRAIN},
                                                transform_param = dict(scale=0.00390625,
                                                                       #crop_size = 32,
                                                                       #mean_file=mean_file, 
                                                                       #mirror=True
                                                                       ))
            first_layer = str(net.to_proto())
            net.data, net.label = caffe.layers.Data(name="face",source=val_lmdb, backend=caffe.params.Data.LMDB, batch_size=val_batch_size, ntop=2,   
                                                image_data_param=dict(shuffle=True),
                                                include={'phase':caffe.TEST},
                                                transform_param = dict(scale=0.00390625,
                                                                       #crop_size = 32, 
                                                                       #mean_file=mean_file, 
                                                                       #mirror=True
                                                                       ))     
######################################
          
        #第二层Convolution视觉层
#        net.conv1 = caffe.layers.Convolution(net.data, num_output=20, kernel_size=3, pad=0,
#                                             param=[dict(lr_mult=1),dict(lr_mult=2)],
#                                             weight_filler={"type": "xavier"},bias_filler={"type": "constant"})
    #if include_type=="deploy":
    net.conv1 = caffe.layers.Convolution(bottom='data', num_output=32, kernel_size=3, stride=1,pad=1,
                                             param=[dict(lr_mult=1),dict(lr_mult=2)],
                                             weight_filler=dict(type='gaussian',std=0.1),bias_filler=dict(type='constant',value=0))
    
    #第三层ReLU激活层
    net.relu1 = caffe.layers.ReLU(net.conv1, in_place=True)
    #第四层Pooling池化层
    net.pool1 = caffe.layers.Pooling(net.relu1, pool=caffe.params.Pooling.MAX, kernel_size=2, stride=2,pad=0)
    # 局部响应归一化
    net.norm1=caffe.layers.LRN(net.pool1,lrn_param = dict(local_size=5,alpha=0.001 / 9.0,beta=0.75))
    
######################################
    net.conv2 = caffe.layers.Convolution(net.norm1, kernel_size=3, stride=1,num_output=64, pad=1,
                                         param=[dict(lr_mult=1),dict(lr_mult=2)],
                                         weight_filler=dict(type='gaussian',std=0.1),bias_filler=dict(type='constant',value=0.1))
    net.relu2 = caffe.layers.ReLU(net.conv2, in_place=True)
    net.pool2 = caffe.layers.Pooling(net.relu2, pool=caffe.params.Pooling.MAX, kernel_size=2, stride=2,pad=0)
    net.norm2=caffe.layers.LRN(net.pool2,lrn_param = dict(local_size=5,alpha=0.001 / 9.0,beta=0.75))
 ######################################
   
    net.conv3 = caffe.layers.Convolution(net.norm2, kernel_size=3, stride=1,num_output=64, pad=1,
                                         param=[dict(lr_mult=1),dict(lr_mult=2)],
                                         weight_filler=dict(type='gaussian',std=0.1),bias_filler=dict(type='constant',value=0.0))
    net.relu3 = caffe.layers.ReLU(net.conv3, in_place=True)
    net.pool3 = caffe.layers.Pooling(net.relu3, pool=caffe.params.Pooling.MAX, kernel_size=2, stride=2,pad=0)  
    net.norm3=caffe.layers.LRN(net.pool3,lrn_param = dict(local_size=5,alpha=0.001 / 9.0,beta=0.75))
######################################
   
    net.conv4 = caffe.layers.Convolution(net.norm3, kernel_size=3, stride=1,num_output=64, pad=1,
                                         param=[dict(lr_mult=1),dict(lr_mult=2)],
                                         weight_filler=dict(type='gaussian',std=0.1),bias_filler=dict(type='constant',value=0.0))
    net.relu4 = caffe.layers.ReLU(net.conv4, in_place=True)
    net.pool4 = caffe.layers.Pooling(net.relu4, pool=caffe.params.Pooling.MAX, kernel_size=2, stride=2,pad=0)  
    net.norm4=caffe.layers.LRN(net.pool4,lrn_param = dict(local_size=3,alpha=0.001 / 9.0,beta=0.75))

######################################
#    net.conv5 = caffe.layers.Convolution(net.norm4, kernel_size=3, stride=1,num_output=64, pad=0,
#                                         param=[dict(lr_mult=1),dict(lr_mult=2)],
#                                         weight_filler=dict(type='gaussian',std=0.1),bias_filler=dict(type='constant',value=0))
#    net.relu5 = caffe.layers.ReLU(net.conv5, in_place=True)
#    net.pool5 = caffe.layers.Pooling(net.relu5, pool=caffe.params.Pooling.MAX, kernel_size=2, stride=2,pad=0)  
#    net.norm5=caffe.layers.LRN(net.pool5,lrn_param = dict(local_size=3,alpha=0.0001,beta=0.75))
#######################################
#    net.conv6 = caffe.layers.Convolution(net.norm5, kernel_size=3, stride=1,num_output=64, pad=0,
#                                         param=[dict(lr_mult=1),dict(lr_mult=2)],
#                                         weight_filler=dict(type='gaussian',std=0.1),bias_filler=dict(type='constant',value=0))
#    net.relu6 = caffe.layers.ReLU(net.conv6, in_place=True)
#    net.pool6 = caffe.layers.Pooling(net.relu6, pool=caffe.params.Pooling.MAX, kernel_size=2, stride=2,pad=0)  
#    net.norm6=caffe.layers.LRN(net.pool6,lrn_param = dict(local_size=5,alpha=0.0001,beta=0.75))    
######################################
   #全连层
    if include_type=="deploy_fc":
        net.fc65_conv = caffe.layers.Convolution(net.norm4, kernel_size=2, num_output=1024, #stride=1,#pad=0,
                                         #param=[dict(lr_mult=1),dict(lr_mult=2)],
                                         #weight_filler=dict(type='xavier'),bias_filler=dict(type='constant')
                                         )
        net.relu65 = caffe.layers.ReLU(net.fc65_conv, in_place=True)        
    else:
        net.fc65 = caffe.layers.InnerProduct(net.norm4, num_output=1024,
                                            weight_filler=dict(type='gaussian',std=1.0/1024.0),bias_filler=dict(type='constant',value=0.1))
        net.relu65 = caffe.layers.ReLU(net.fc65, in_place=True)
        net.drop65 = caffe.layers.Dropout(net.relu65 ,in_place=True,dropout_param  = dict(dropout_ratio=0.1))
######################################
#   #全连层
#    if include_type=="deploy_fc":
#        net.fc66_conv = caffe.layers.Convolution(net.relu65, kernel_size=1, num_output=1024, #stride=1,#pad=0,
#                                         #param=[dict(lr_mult=1),dict(lr_mult=2)],
#                                         #weight_filler=dict(type='xavier'),bias_filler=dict(type='constant')
#                                         )
#        net.relu66 = caffe.layers.ReLU(net.fc66_conv, in_place=True)        
#    else:
#        net.fc66 = caffe.layers.InnerProduct(net.drop65, num_output=1024,
#                                            weight_filler=dict(type='gaussian',std=1.0/1024.0),bias_filler=dict(type='constant',value=0.1))
#        net.relu66= caffe.layers.ReLU(net.fc66, in_place=True)
#        net.drop66 = caffe.layers.Dropout(net.relu66 ,in_place=True,dropout_param  = dict(dropout_ratio=0.5))
######################################
   #创建一个dropout层
    if include_type=="deploy_fc":
        #net.drop3 = caffe.layers.Dropout(net.relu5, in_place=True)
        net.fc100_conv = caffe.layers.Convolution(net.relu65 , kernel_size=1, num_output=62,#stride=1, #pad=0,
                                         #param=[dict(lr_mult=1),dict(lr_mult=2)],
                                         #weight_filler=dict(type='xavier'),bias_filler=dict(type='constant')
                                         )
    else:
        net.fc100 = caffe.layers.InnerProduct(net.drop65, num_output=62,
                                            weight_filler=dict(type='gaussian',std=0.1),bias_filler=dict(type='constant',value=0))
    #创建一个softmax层
    if not include_type=="deploy":
        if not include_type=="deploy_fc":
            net.loss = caffe.layers.SoftmaxWithLoss(net.fc100, net.label)

    if include_type=="deploy":
        net.prob=caffe.layers.Softmax(net.fc100)
        return str(net.to_proto())
    if include_type=="deploy_fc":
        net.prob=caffe.layers.Softmax(net.fc100_conv)
        return str(net.to_proto())    
        
    net.acc = caffe.layers.Accuracy(net.fc100, net.label,include={'phase':caffe.TEST})
    return first_layer+str(net.to_proto())
def write_net():
    caffe_root = root_str    #my-caffe-project目录
    train_lmdb = caffe_root + "train_lmdb"                            #train.lmdb文件的位置
    val_lmdb = caffe_root + "val_lmdb"                            #train.lmdb文件的位置
    mean_file = caffe_root + "imagenet_mean.binaryproto"                     #均值文件的位置
    train_proto = caffe_root + "train_val.prototxt"                        #保存train_prototxt文件的位置
    #val_proto = caffe_root + "val.prototxt"                        #保存test_prototxt文件的位置
    #写入prototxt文件
    with open(train_proto, 'w') as f:
        f.write(str(create_net(train_lmdb,val_lmdb, mean_file, train_batch_size=128,val_batch_size=100)))

#    #写入prototxt文件
#    with open(val_proto, 'w') as f:
#        f.write(str(create_net(val_lmdb, mean_file, batch_size=50, include_type="val")))


def write_sovler():
    my_project_root = root_str        #my-caffe-project目录
    sovler_string = caffe.proto.caffe_pb2.SolverParameter()                    #sovler存储
    solver_file = my_project_root + 'solver.prototxt'                        #sovler文件保存位置
    sovler_string.net = my_project_root + 'train_val.prototxt'
#    sovler_string.train_net = my_project_root + 'train.prototxt'            #train.prototxt位置指定
#    sovler_string.test_net.append(my_project_root + 'val.prototxt')         #test.prototxt位置指定
    sovler_string.test_iter.append(10)                                        #测试迭代次数
    sovler_string.test_interval = 1000                                        #每训练迭代test_interval次进行一次测试
    sovler_string.base_lr = 0.003                                            #基础学习率   
    sovler_string.momentum = 0.9                                            #动量
    sovler_string.weight_decay = 0.005                                        #权重衰减
    
    # inv':return base_lr * (1 + gamma * iter) ^ (- power)
    sovler_string.lr_policy = 'inv'                                        #学习策略      '
    sovler_string.gamma = 0.0002                                          
    sovler_string.power = 0.95                                          
    sovler_string.stepsize = 100000                                     # 当迭代到第一个stepsize次时,lr第一次衰减，衰减后的lr=lr*gamma
    
    sovler_string.display = 200                                                #每迭代display次显示结果
    sovler_string.max_iter = 50000                                            #最大迭代数
    sovler_string.snapshot = 10000                                             #保存临时模型的迭代数
    #sovler_string.snapshot_format = 0                                        #临时模型的保存格式,0代表HDF5,1代表BINARYPROTO
    sovler_string.snapshot_prefix = root_str+'caffe_ljftest_train'        #模型前缀
    sovler_string.solver_mode = caffe.proto.caffe_pb2.SolverParameter.GPU    #优化模式

    with open(solver_file, 'w') as f:
        f.write(str(sovler_string))  
 

def write_deploy(): 
    deploy_root=root_str+'deploy.prototxt'    #文件保存路径

    with open(deploy_root, 'w') as f:
        f.write('name:"Lenet"\n')
        f.write('input:"data"\n')
        f.write('input_dim:1\n')
        f.write('input_dim:3\n')
        f.write('input_dim:32\n')
        f.write('input_dim:32\n')
        f.write(str(create_net(train_lmdb=None,val_lmdb=None,mean_file=None,include_type="deploy")))
        
def write_deploy_fc():     # 全连接转全卷积
    deploy_root=root_str+'deploy_fc.prototxt'    #文件保存路径

    with open(deploy_root, 'w') as f:
        f.write('name:"Lenet"\n')
        f.write('input:"data"\n')
        f.write('input_dim:1\n')
        f.write('input_dim:3\n')
        f.write('input_dim:32\n')
        f.write('input_dim:32\n')
        f.write(str(create_net(train_lmdb=None,val_lmdb=None,mean_file=None,include_type="deploy_fc")))        
        
        
if __name__ == '__main__':
    write_net()
    write_sovler()
    write_deploy()
    write_deploy_fc()
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    