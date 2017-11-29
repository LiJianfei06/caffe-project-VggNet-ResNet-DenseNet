# -*- coding: UTF-8 -*-
import sys
sys.path.append("/home/ljf/caffe-master/python")
sys.path.append("/home/ljf/caffe-master/python/caffe")
import caffe     
from caffe import layers as L,params as P,to_proto
                                                #导入caffe包

root_str="/home/ljf/caffe-master/examples/ljftest_cifar10_mobilenet/"


def conv_BN_scale_relu(split, bottom, nout, ks, stride, pad,group=1): #对输入作 conv—BN—scale—relu
    if bottom=="data":
        conv = L.Convolution(bottom="data", kernel_size = ks, stride = stride, num_output = nout,
                             pad = pad, bias_term = True,
                             weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    else:
        if group!=1:
            conv = L.Convolution(bottom, kernel_size = ks, stride = stride, group=group,num_output = nout,
                                 pad = pad, bias_term = True,
                                 weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        else:
            conv = L.Convolution(bottom, kernel_size = ks, stride = stride,num_output = nout,
                                 pad = pad, bias_term = True,
                                 weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))                
    if split == 'train':
        use_global_stats = False #训练的时候我们对 BN 的参数取滑动平均
    else:
        use_global_stats = True #测试的时候我们直接使用输入的参数
    BN = L.BatchNorm(conv, batch_norm_param = dict(use_global_stats = use_global_stats),
                     in_place = True, #BN 的学习率惩罚设置为 0，由 scale 学习
#                     param = [dict(lr_mult = 0, decay_mult = 0),
#                              dict(lr_mult = 0, decay_mult = 0),
#                              dict(lr_mult = 0, decay_mult = 0)]
                     )
    scale = L.Scale(BN, scale_param = dict(bias_term = True), in_place = True)
    relu = L.ReLU(scale, in_place = True)
    return scale, relu







def mobilenet(split):
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
        
    
    if split == 'deploy':
        scale, result = conv_BN_scale_relu(split, bottom="data", nout = 32, ks = 3, stride = 1, pad = 1,group=1) #conv1
    else:
        scale, result = conv_BN_scale_relu(split, bottom=data, nout = 32, ks = 3, stride = 1, pad = 1,group=1) #conv1

    
    scale,result = conv_BN_scale_relu(split, bottom=result, nout = 32, ks = 3, stride = 1, pad = 1,group=32) #conv1
    scale,result = conv_BN_scale_relu(split, bottom=result, nout = 64, ks = 1, stride = 1, pad = 0,group=1) #conv1
    
    scale,result = conv_BN_scale_relu(split, bottom=result, nout = 64, ks = 3, stride = 2, pad = 1,group=64) #conv1
    scale,result = conv_BN_scale_relu(split, bottom=result, nout = 128, ks = 1, stride = 1, pad = 0,group=1) #conv1
    
    scale,result = conv_BN_scale_relu(split, bottom=result, nout = 128, ks = 3, stride = 2, pad = 1,group=128) #conv1
    scale,result = conv_BN_scale_relu(split, bottom=result, nout = 256, ks = 1, stride = 1, pad = 0,group=1) #conv1
      
    scale,result = conv_BN_scale_relu(split, bottom=result, nout = 256, ks = 3, stride = 2, pad = 1,group=256) #conv1
    scale,result = conv_BN_scale_relu(split, bottom=result, nout = 512, ks = 1, stride = 1, pad = 0,group=1) #conv1

    scale,result = conv_BN_scale_relu(split, bottom=result, nout = 512, ks = 3, stride = 2, pad = 1,group=512) #conv1
    scale,result = conv_BN_scale_relu(split, bottom=result, nout = 1024, ks = 1, stride = 1, pad = 0,group=1) #conv1
    
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
    sovler_string.test_interval = 1000                                        #每训练迭代test_interval次进行一次测试
    sovler_string.iter_size=1
    sovler_string.base_lr = 0.1                                            #基础学习率   
    sovler_string.momentum = 0.9                                            #动量
    sovler_string.weight_decay = 1e-4                                        #权重衰减
    sovler_string.type = 'Nesterov'
    # inv':return base_lr * (1 + gamma * iter) ^ (- power)
    sovler_string.lr_policy = 'multistep'                                        #学习策略      '
    sovler_string.gamma = 0.1                                          
   # sovler_string.power = 0.95                                          
    sovler_string.stepsize = 200000                                     # 当迭代到第一个stepsize次时,lr第一次衰减，衰减后的lr=lr*gamma
    
    sovler_string.display = 200                                                #每迭代display次显示结果
    sovler_string.max_iter = 100000                                            #最大迭代数
    sovler_string.snapshot = 2500                                             #保存临时模型的迭代数
    
    sovler_string.stepvalue.append(36000)
    sovler_string.stepvalue.append(50000)
    sovler_string.stepvalue.append(72000)
    sovler_string.stepvalue.append(90000)         
    #sovler_string.snapshot_format = 0                                        #临时模型的保存格式,0代表HDF5,1代表BINARYPROTO
    sovler_string.snapshot_prefix = root_str+'model_save/caffe_ljftest_train'        #模型前缀


    sovler_string.solver_mode = caffe.proto.caffe_pb2.SolverParameter.GPU    #优化模式

    with open(solver_file, 'w') as f:
        f.write(str(sovler_string))  
 

def write_deploy(): 
    deploy_root=root_str+'deploy.prototxt'    #文件保存路径

    with open(deploy_root, 'w') as f:
        f.write('name:"mobilenet"\n')
        f.write('input:"data"\n')
        f.write('input_dim:1\n')
        f.write('input_dim:3\n')
        f.write('input_dim:28\n')
        f.write('input_dim:28\n')
        f.write(str(mobilenet("deploy")))
        
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
        f.write(str(mobilenet('train')))#创建 train.prototxt
    with open(root_str + "test.prototxt", 'w') as f:
        f.write(str(mobilenet('test')))#创建 train.prototxt
    
    write_deploy()
        
    write_sovler()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    