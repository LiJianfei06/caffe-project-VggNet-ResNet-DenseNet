from __future__ import print_function

import sys
                                       

sys.path.append("/home/ljf/caffe-master/python")
sys.path.append("/home/ljf/caffe-master/python/caffe")

from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import caffe


root_str="/home/ljf/caffe-master/examples/ljftest_cifar10_DenseNet/"




def bn_relu_conv(bottom, ks, nout, stride, pad, dropout):
    conv1=L.Convolution(bottom, kernel_size=1, stride=1, 
                        num_output=nout,pad=0, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant')) 
    conv = L.Convolution(conv1, kernel_size=ks, stride=stride, 
                         num_output=nout, pad=pad, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant')) 
    batch_norm = L.BatchNorm(conv, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
    relu = L.ReLU(scale, in_place=True)

    
#    if dropout>0:
#        relu = L.Dropout(relu, dropout_ratio=dropout)
    return relu

def add_layer(bottom, num_filter, dropout):
     
    conv = bn_relu_conv(bottom, ks=3, nout=num_filter, stride=1, pad=1, dropout=dropout)
    
    concate = L.Concat(bottom, conv, axis=1)
    return concate

def transition(bottom, num_filter, dropout):
  
    conv = bn_relu_conv(bottom, ks=1, nout=num_filter, stride=1, pad=0, dropout=dropout)
    pooling = L.Pooling(conv, pool=P.Pooling.AVE, kernel_size=2, stride=2)
    return pooling

#change the line below to experiment with different setting
#depth -- must be 3n+4
#first_output -- #channels before entering the first dense block, set it to be comparable to growth_rate
#growth_rate -- growth rate
#dropout -- set to 0 to disable dropout, non-zero number to set dropout rate
def densenet(data_file=None, mode='train', batch_size=64, depth=20, first_output=32, growth_rate=32, dropout=0.5):
    if mode=='train':
        data, label = L.Data(source=data_file, backend=P.Data.LMDB, batch_size=batch_size, ntop=2, 
                             image_data_param=dict(shuffle=True),
                  transform_param=dict(#mean_file="/home/ljf/caffe-master/examples/ljftest_alphabet_DenseNet/imagenet_mean.binaryproto"
                                       crop_size = 28,
                                       #scale=0.00390625,
                                       mirror=True
                                       ))
    if mode=='test':
         data, label = L.Data(source=data_file, backend=P.Data.LMDB, batch_size=batch_size, ntop=2, 
                         #image_data_param=dict(shuffle=True),
                         transform_param=dict(#mean_file="/home/ljf/caffe-master/examples/ljftest_alphabet_DenseNet/imagenet_mean.binaryproto"
                                   crop_size = 28,
                                   #scale=0.00390625,
                                   #mirror=True
                                   ))
        

    nchannels = first_output
    if mode == 'deploy':
        model = L.Convolution(bottom="data", kernel_size=3, stride=1, num_output=nchannels,
                        pad=1, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    else:
        model = L.Convolution(data, kernel_size=3, stride=1, num_output=nchannels,
                        pad=1, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))

    #N = (depth-4)/4
    N=3
    for i in range(N):
        model = add_layer(model, growth_rate, dropout)
        nchannels += growth_rate
    model = transition(model, nchannels, dropout)
    
    N=3
    for i in range(N):
        model = add_layer(model, growth_rate, dropout)
        nchannels += growth_rate
    model = transition(model, nchannels, dropout)

    N=3
    for i in range(N):
        model = add_layer(model, growth_rate, dropout)
        nchannels += growth_rate
    model = transition(model, nchannels, dropout)

    N=3
    for i in range(N):
        model = add_layer(model, growth_rate, dropout)
        nchannels += growth_rate
    model = transition(model, nchannels, dropout)    
#    N=7
#    for i in range(N):
#        model = add_layer(model, growth_rate, dropout)
#        nchannels += growth_rate



    model = L.BatchNorm(model, in_place=False, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    model = L.Scale(model, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
    model = L.ReLU(model, in_place=True)
    model = L.Pooling(model, pool=P.Pooling.AVE, global_pooling=True)
    model = L.InnerProduct(model, num_output=10, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
    
    if mode == 'deploy':
        prob=L.Softmax(model)
        return to_proto(prob)   
    else:
        loss = L.SoftmaxWithLoss(model, label)    
        if mode == 'train':
                return to_proto(loss)
        accuracy = L.Accuracy(model, label)
        return to_proto(loss, accuracy)

def make_net():

    with open('train.prototxt', 'w') as f:
        #change the path to your data. If it's not lmdb format, also change first line of densenet() function
        print(str(densenet(root_str+'train_lmdb', mode='train',batch_size=16)), file=f)

    with open('test.prototxt', 'w') as f:
        print(str(densenet(root_str+'test_lmdb',mode='test', batch_size=10)), file=f)

def make_solver():
    s = caffe_pb2.SolverParameter()
    s.random_seed = 0xCAFFE

    s.train_net = root_str+'train.prototxt'
    s.test_net.append(root_str+'test.prototxt')
    s.test_interval = 1000
    s.test_iter.append(1000)
    s.iter_size = 8 
    s.max_iter = 100000
    s.type = 'Nesterov'
    s.display = 200

    s.base_lr = 0.1
    s.momentum = 0.9
    s.weight_decay = 1e-4
    s.snapshot_prefix = root_str+'model_save/caffe_ljftest_train'
    s.lr_policy='multistep'
    s.gamma = 0.1
    s.snapshot=5000
    s.stepvalue.append(32000)
    s.stepvalue.append(48000)
    s.stepvalue.append(72000)
    s.stepvalue.append(96000)
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    solver_path = 'solver.prototxt'
    with open(solver_path, 'w') as f:
        f.write(str(s))

def write_deploy(): 
    deploy_root=root_str+'deploy.prototxt'

    with open(deploy_root, 'w') as f:
        f.write('name:"DenseNet"\n')
        f.write('input:"data"\n')
        f.write('input_dim:1\n')
        f.write('input_dim:3\n')
        f.write('input_dim:28\n')
        f.write('input_dim:28\n')
        #f.write(str(ResNet("deploy")))
        print(str(densenet(mode='deploy')), file=f)
        
        
if __name__ == '__main__':

    make_net()
    make_solver()
    write_deploy()









