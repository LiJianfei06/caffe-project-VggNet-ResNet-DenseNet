# -*- coding: UTF-8 -*-
import sys
sys.path.append("/home/ljf/caffe-master/python")
sys.path.append("/home/ljf/caffe-master/python/caffe")
import caffe     
from caffe import layers as L,params as P,to_proto
                                                #导入caffe包

root_str="/home/ljf/caffe-master/examples/ljftest_cifar10_ResNeXt/"


def conv_BN_scale_relu(split, bottom, nout, ks, stride, pad): #对输入作 conv—BN—scale—relu
    if bottom=="data":
        conv = L.Convolution(bottom="data", kernel_size = ks, stride = stride, num_output = nout,
                             pad = pad, bias_term = True,
                             weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    else:
        conv = L.Convolution(bottom, kernel_size = ks, stride = stride, num_output = nout,
                             pad = pad, bias_term = True,
                             weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
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

def ResNet_block1(split, bottom, nout, ks, stride, projection_stride, pad):

    scale0, relu0 = conv_BN_scale_relu(split, bottom, nout, ks, projection_stride, pad)
    scale3, relu3 = conv_BN_scale_relu(split, bottom, nout, ks, projection_stride, pad)
        
    scale1, relu1 = conv_BN_scale_relu(split, bottom, nout, ks, projection_stride, pad)
    scale2, relu2 = conv_BN_scale_relu(split, relu1, nout, ks, stride, pad)
    wise = L.Eltwise(scale2, scale0, scale0,operation = P.Eltwise.SUM) #数据相加
    wise_relu = L.ReLU(wise, in_place = True)
    return wise_relu

def ResNet(split):
    train_data_file = root_str + 'train_lmdb'
    test_data_file = root_str + 'test_lmdb'
    #mean_file = root_str + 'imagenet_mean.binaryproto'


    if split == 'train':
        data, labels = L.Data(source = train_data_file,
                              backend = P.Data.LMDB,
                              ntop = 2,
                              batch_size = 128,
                              image_data_param=dict(shuffle=True),
                                                    #include={'phase':caffe.TRAIN},
                                                    transform_param = dict(#scale=0.00390625,
                                                                           crop_size = 28,
                                                                           #mean_file=mean_file, 
                                                                           mirror=True
                                                                           ))
    else:
        data, labels = L.Data(source = test_data_file,
                              backend = P.Data.LMDB,
                              ntop = 2,
                              batch_size = 100,
                              image_data_param=dict(shuffle=True),
                                                    #include={'phase':caffe.TRAIN},
                                                    transform_param = dict(#scale=0.00390625,
                                                                           crop_size = 28,
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
        
        
        
        
        
        
        
        
        
        
        

        
    repeat = 3                          #每个 ConX_X 都有 3 个 Residual Block
    
    if split == 'deploy':
        scale, result = conv_BN_scale_relu(split, bottom="data", nout = 32, ks = 3, stride = 1, pad = 1) #conv1
    else:
        scale, result = conv_BN_scale_relu(split, bottom=data, nout = 32, ks = 3, stride = 1, pad = 1) #conv1

    
    repeat = 3 
    result1=result
    for i in range(repeat):             #conv2_x
        projection_stride = 1           #输入与输出的数据通道数都是 16，大小都是 32 X 32
                                        #可以直接相加，设置映射步长为 1
        result1 = ResNet_block(split, result1, nout = 32, ks = 3, stride = 1, projection_stride =
                              projection_stride, pad = 1)
        
    result2=result
    for i in range(repeat):             #conv2_x
        projection_stride = 1           #输入与输出的数据通道数都是 16，大小都是 32 X 32
                                        #可以直接相加，设置映射步长为 1
        result2 = ResNet_block(split, result2, nout = 32, ks = 3, stride = 1, projection_stride =
                              projection_stride, pad = 1)        
    result3=result
    for i in range(repeat):             #conv2_x
        projection_stride = 1           #输入与输出的数据通道数都是 16，大小都是 32 X 32
                                        #可以直接相加，设置映射步长为 1
        result3 = ResNet_block(split, result3, nout = 32, ks = 3, stride = 1, projection_stride =
                              projection_stride, pad = 1)

    result4=result
    for i in range(repeat):             #conv2_x
        projection_stride = 1           #输入与输出的数据通道数都是 16，大小都是 32 X 32
                                        #可以直接相加，设置映射步长为 1
        result4 = ResNet_block(split, result4, nout = 32, ks = 3, stride = 1, projection_stride =
                              projection_stride, pad = 1)

    result5=result
    for i in range(repeat):             #conv2_x
        projection_stride = 1           #输入与输出的数据通道数都是 16，大小都是 32 X 32
                                        #可以直接相加，设置映射步长为 1
        result5 = ResNet_block(split, result5, nout = 32, ks = 3, stride = 1, projection_stride =
                              projection_stride, pad = 1)
        
    result = L.Eltwise(result1, result2,result3,result4,result5,operation = P.Eltwise.SUM)
        
        
        
     
    repeat = 3
    for i in range(repeat): #conv3_x
        if i == 0: #只有在刚开始 conv2_x(16 X 16)到 conv3_x(8 X 8)的数据维度不一样，需要映射到相同维度
            projection_stride = 2 # 卷积映射的 stride 为 2
        else:
            projection_stride = 1 # 卷积映射的 stride 为 2
        result = ResNet_block(split, result, nout = 64, ks = 3, stride = 1, projection_stride
                              =projection_stride, pad = 1)







    pool = L.Pooling(result, pool = P.Pooling.AVE, global_pooling = True)
    #pool = L.Pooling(result, pool=P.Pooling.AVE, kernel_size=4, stride=1,pad=0)  
    IP = L.InnerProduct(pool, num_output = 10,
                        weight_filler=dict(type='xavier'),bias_filler=dict(type='constant'))
    
    
    
    
    
    
    
    
    
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
    sovler_string.random_seed = 0xCAFFE
    solver_file = my_project_root + 'solver.prototxt'                        #sovler文件保存位置
    #sovler_string.net = my_project_root + 'train_val.prototxt'
    sovler_string.train_net = my_project_root + 'train.prototxt'            #train.prototxt位置指定
    sovler_string.test_net.append(my_project_root + 'test.prototxt')         #test.prototxt位置指定
    sovler_string.test_iter.append(100)                                        #测试迭代次数
    sovler_string.test_interval = 100                                        #每训练迭代test_interval次进行一次测试
    sovler_string.base_lr = 0.1                                            #基础学习率   
    sovler_string.momentum = 0.9                                            #动量
    sovler_string.weight_decay = 1e-4                                        #权重衰减
    sovler_string.type = 'Nesterov'
    # inv':return base_lr * (1 + gamma * iter) ^ (- power)
    sovler_string.lr_policy = 'multistep'                                        #学习策略      '
    sovler_string.gamma = 0.1                                          
   # sovler_string.power = 0.95                                          
    sovler_string.stepsize = 200000                                     # 当迭代到第一个stepsize次时,lr第一次衰减，衰减后的lr=lr*gamma
    
    sovler_string.display = 20                                                #每迭代display次显示结果
    sovler_string.max_iter = 200000                                            #最大迭代数
    sovler_string.snapshot = 10000                                             #保存临时模型的迭代数
    
    sovler_string.stepvalue.append(int(0.02 * sovler_string.max_iter))
    sovler_string.stepvalue.append(int(0.2 * sovler_string.max_iter))
    sovler_string.stepvalue.append(int(0.45 * sovler_string.max_iter))
    sovler_string.stepvalue.append(int(0.75 * sovler_string.max_iter))    
    #sovler_string.snapshot_format = 0                                        #临时模型的保存格式,0代表HDF5,1代表BINARYPROTO
    sovler_string.snapshot_prefix = root_str+'model_save/caffe_ljftest_train'        #模型前缀


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
        f.write('input_dim:28\n')
        f.write('input_dim:28\n')
        f.write(str(ResNet("deploy")))
        
#def write_deploy_fc():     # 全连接转全卷积
#    deploy_root=root_str+'deploy_fc.prototxt'    #文件保存路径
#
#    with open(deploy_root, 'w') as f:
#        f.write('name:"Lenet"\n')
#        f.write('input:"data"\n')
#        f.write('input_dim:1\n')
#        f.write('input_dim:3\n')
#        f.write('input_dim:28\n')
#        f.write('input_dim:28\n')
#        f.write(str(create_net(file_lmdb=None,mean_file=None,include_type="deploy_fc")))        
#        
        
if __name__ == '__main__':
    with open(root_str + "train.prototxt", 'w') as f:
        f.write(str(ResNet('train')))#创建 train.prototxt
    with open(root_str + "test.prototxt", 'w') as f:
        f.write(str(ResNet('test')))#创建 train.prototxt
    
    write_deploy()
        
    write_sovler()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    