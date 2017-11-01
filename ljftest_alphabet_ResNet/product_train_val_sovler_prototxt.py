# -*- coding: UTF-8 -*-
import sys
                                       

sys.path.append("/home/ljf/caffe-master/python")
sys.path.append("/home/ljf/caffe-master/python/caffe")
import caffe     
from caffe import layers as L,params as P,to_proto


root_str="/home/ljf/caffe-master/examples/ljftest_alphabet_ResNet/"

import tools
#import read_data




def conv_BN_scale_relu(split, bottom, nout, ks, stride, pad): #对输入作 conv—BN—scale—relu
    if bottom=="data":
        conv = L.Convolution(bottom="data", kernel_size = ks, stride = stride, num_output = nout,
                             pad = pad, bias_term = True,
                             weight_filler = dict(type='gaussian',std=0.1),
                             bias_filler = dict(type = 'constant'))
    else:
        conv = L.Convolution(bottom, kernel_size = ks, stride = stride, num_output = nout,
                             pad = pad, bias_term = True,
                             weight_filler = dict(type='gaussian',std=0.1),
                             bias_filler = dict(type = 'constant'))        
    if split == 'train':
        use_global_stats = False #训练的时候我们对 BN 的参数取滑动平均
    else:
        use_global_stats = True #测试的时候我们直接使用输入的参数
    BN = L.BatchNorm(conv, batch_norm_param = dict(use_global_stats = use_global_stats),
                     in_place = True, #BN 的学习率惩罚设置为 0，由 scale 学习
                     param = [dict(lr_mult = 0, decay_mult = 0),
                              dict(lr_mult = 0, decay_mult = 0),
                              dict(lr_mult = 0, decay_mult = 0)])
    scale = L.Scale(BN, scale_param = dict(bias_term = True), in_place = True)
    relu = L.ReLU(scale, in_place = True)
    return scale, relu






def ResNet_block(split, bottom, nout, ks, stride, projection_stride, pad):
    if projection_stride == 1: #1 代表不需要 1X1 的映射
        scale0 = bottom
    else: #否则经过 1X1，stride=2 的映射
        scale0, relu0 = conv_BN_scale_relu(split, bottom, nout, 1, projection_stride, 0)
        
    scale1, relu1 = conv_BN_scale_relu(split, bottom, nout, ks, projection_stride, pad)
    scale2, relu2 = conv_BN_scale_relu(split, relu1, nout, ks, stride, pad)
    wise = L.Eltwise(scale2, scale0, operation = P.Eltwise.SUM) #数据相加
    wise_relu = L.ReLU(wise, in_place = True)
    return wise_relu



def ResNet(split):
    train_data_file = root_str + 'train_lmdb'
    test_data_file = root_str + 'val_lmdb'
    mean_file = root_str + 'imagenet_mean.binaryproto'


    if split == 'train':
        data, labels = L.Data(source = train_data_file,
                              backend = P.Data.LMDB,
                              ntop = 2,
                              batch_size = 256,
                              image_data_param=dict(shuffle=True),
                                                    #include={'phase':caffe.TRAIN},
                                                    transform_param = dict(#scale=0.00390625,
                                                                           #crop_size = 28,
                                                                           #mean_file=mean_file, 
                                                                           #mirror=True
                                                                           ))
    else:
        data, labels = L.Data(source = test_data_file,
                              backend = P.Data.LMDB,
                              ntop = 2,
                              batch_size = 100,
                              image_data_param=dict(shuffle=True),
                                                    #include={'phase':caffe.TRAIN},
                                                    transform_param = dict(#scale=0.00390625,
                                                                           #crop_size = 28,
                                                                           #mean_file=mean_file, 
                                                                           #mirror=True
                                                                           ))
  
#    data, labels = L.Python(module = 'read_data', #文件名
#                            layer = 'input_layer',#class 的名字
#                            ntop = 2,#输出个数,data 和 labels
#                            param_str = str(dict(split = split,
#                                                 #设置数据路径
#                                                 data_dir = "data/cifar10/",
#                                                 #设置训练数据名字
#                                                 train_data_name = 'train_data.h5',
#                                                 #设置测试数据名字
#                                                 test_data_name = 'test_data.h5',
#                                                 #设置训练数据 batch
#                                                 train_batches = 128,
#                                                 #设置训练数据 batch
#                                                 test_batches = 100,
#                                                 pad_size = 4#设置 pad 大小
#                                                 )))        
        
        
        
        
        
        
        
        
        
        
        

        
    repeat = 4                          #每个 ConX_X 都有 3 个 Residual Block
    
    if split == 'deploy':
        scale, result = conv_BN_scale_relu(split, bottom="data", nout = 16, ks = 3, stride = 1, pad = 1) #conv1
    else:
        scale, result = conv_BN_scale_relu(split, bottom=data, nout = 16, ks = 3, stride = 1, pad = 1) #conv1

    
    for i in range(repeat):             #conv2_x
        projection_stride = 1           #输入与输出的数据通道数都是 16，大小都是 32 X 32
                                        #可以直接相加，设置映射步长为 1
        result = ResNet_block(split, result, nout = 16, ks = 3, stride = 1, projection_stride =
                              projection_stride, pad = 1)
     
    repeat = 5
    for i in range(repeat): #conv3_x
        if i == 0: #只有在刚开始 conv2_x(16 X 16)到 conv3_x(8 X 8)的数据维度不一样，需要映射到相同维度
            projection_stride = 2 # 卷积映射的 stride 为 2
        else:
            projection_stride = 1 # 卷积映射的 stride 为 2
        result = ResNet_block(split, result, nout = 32, ks = 3, stride = 1, projection_stride
                              =projection_stride, pad = 1)
    repeat = 5    
    for i in range(repeat): #conv4_x
        if i == 0:
            projection_stride = 2
        else:
            projection_stride = 1
        result = ResNet_block(split, result, nout = 64, ks = 3, stride = 1, projection_stride
                              =projection_stride, pad = 1)
        
    repeat = 4
    for i in range(repeat): #conv4_x
        if i == 0:
            projection_stride = 2
        else:
            projection_stride = 1
        result = ResNet_block(split, result, nout = 128, ks = 3, stride = 1, projection_stride
                              =projection_stride, pad = 1)


#    repeat = 4
#    for i in range(repeat): #conv4_x
#        if i == 0:
#            projection_stride = 2
#        else:
#            projection_stride = 1
#        result = ResNet_block(split, result, nout = 256, ks = 3, stride = 1, projection_stride
#                              =projection_stride, pad = 1)






    pool = L.Pooling(result, pool = P.Pooling.AVE, global_pooling = True)
    IP = L.InnerProduct(pool, num_output = 62,
                        weight_filler = dict(type='gaussian',std=0.1),
                        bias_filler = dict(type = 'constant'))
    
    
    
    
    
    
    
    
    
    if split == 'deploy':
        prob=L.Softmax(IP)
        return to_proto(prob)
    else:
        loss = L.SoftmaxWithLoss(IP, labels)
        if split == 'train':
            return to_proto(loss)
         
        acc = L.Accuracy(IP, labels)
        return to_proto(acc, loss)






def write_sovler():
    my_project_root = root_str        #my-caffe-project目录
    sovler_string = caffe.proto.caffe_pb2.SolverParameter()                    #sovler存储
    solver_file = my_project_root + 'solver.prototxt'                        #sovler文件保存位置
    #sovler_string.net = my_project_root + 'train_val.prototxt'
    sovler_string.train_net = my_project_root + 'train.prototxt'            #train.prototxt位置指定
    sovler_string.test_net.append(my_project_root + 'test.prototxt')         #test.prototxt位置指定
    sovler_string.test_iter.append(100)                                        #测试迭代次数
    sovler_string.test_interval = 100                                        #每训练迭代test_interval次进行一次测试
    sovler_string.base_lr = 0.1                                            #基础学习率   
    sovler_string.momentum = 0.9                                            #动量
    sovler_string.weight_decay = 0.0005                                        #权重衰减
    
    # inv':return base_lr * (1 + gamma * iter) ^ (- power)
    sovler_string.lr_policy = 'multistep'                                        #学习策略      '
    sovler_string.gamma = 0.1                                          
    #sovler_string.power = 0.95                                          
#    sovler_string.stepvalue = 1000                                     # 当迭代到第一个stepsize次时,lr第一次衰减，衰减后的lr=lr*gamma
#    sovler_string.stepvalue1 = 2000    
#    sovler_string.stepvalue2 = 3000    
#    sovler_string.stepvalue3= 4000    
#    sovler_string.stepvalue4 = 5000    
    sovler_string.display = 20                                                #每迭代display次显示结果
    sovler_string.max_iter = 100000                                            #最大迭代数
    sovler_string.snapshot = 2000                                             #保存临时模型的迭代数
    #sovler_string.snapshot_format = 0                                        #临时模型的保存格式,0代表HDF5,1代表BINARYPROTO
    sovler_string.snapshot_prefix = root_str+'caffe_ljftest_train'        #模型前缀
    sovler_string.solver_mode = caffe.proto.caffe_pb2.SolverParameter.GPU    #优化模式

    with open(solver_file, 'w') as f:
        f.write(str(sovler_string))  
 






def write_deploy(): 
    deploy_root=root_str+'deploy.prototxt'    #文件保存路径

    with open(deploy_root, 'w') as f:
        f.write('name:"ResNet"\n')
        f.write('input:"data"\n')
        f.write('input_dim:1\n')
        f.write('input_dim:3\n')
        f.write('input_dim:32\n')
        f.write('input_dim:32\n')
        f.write(str(ResNet("deploy")))
        
        
        
        
        
        
if __name__ == '__main__':
    with open(root_str + "train.prototxt", 'w') as f:
        f.write(str(ResNet('train')))#创建 train.prototxt
    with open(root_str + "test.prototxt", 'w') as f:
        f.write(str(ResNet('test')))#创建 train.prototxt
    
    write_deploy()
#      
    solver_dir = root_str + 'solver.prototxt'
    solver_prototxt = tools.CaffeSolver()
    solver_prototxt.write(solver_dir)
    #把内容写入 res_net_model 文件夹中的 res_net_solver.prototxt
    #write_sovler()



    
    
 
    
    
    
    
    
    
    
    
    
    