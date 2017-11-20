============准备============：
1首先，准备数据包：train val 文件夹 train.txt val.txt文件 
2在/home/ljf/caffe-master目录下执行 ./examples/ljftest_cifar10_DenseNet/create_imagenet.sh 得到两个数据集： train_lmdb和test_lmdb   // cifar10不用这样
3在/home/ljf/caffe-master目录下执行./examples/ljftest_cifar10_DenseNet/make_imagenet_mean.sh
4在/home/ljf/caffe-master目录下执行python examples/ljftest_cifar10_DenseNet/convert_mean.py examples/ljftest_cifar10_DenseNet/imagenet_mean.binaryproto examples/ljftest_cifar10_DenseNet/imagenet_mean.npy


============训练============:
1在/home/ljf/caffe-master目录下执行python examples/ljftest_cifar10_DenseNet/product_train_val_sovler_prototxt.py
生成train_val.prototxt ; solver.prototxt ; deploy.prototxt ; deploy_fc.prototxt
2在/home/ljf/caffe-master目录下执行 ./examples/ljftest_cifar10_VggNet/train_net.sh 开始训练caffe.log会记录全部打印内容
 或 python ./examples/ljftest_cifar10_DenseNet/run.py >>./examples/ljftest_cifar10_DenseNet/caffe_python.log


============测试============：
1在/home/ljf/caffe-master目录下执行./examples/ljftest_cifar10_DenseNet/test_result.sh  测试50个batch
2在/home/ljf/caffe-master目录下执行 python ./examples/ljftest_cifar10_DenseNet/verify.py  测试自定义个batch

测试一张图片：
1准备好labels.txt和图片文件
2在/home/ljf/caffe-master目录下执行./examples/ljftest_cifar10_DenseNet/test_a_image.sh

python接口使用模型测试：
在/home/ljf/caffe-master目录下执行: python ./examples/ljftest_cifar10_DenseNet/forecast_Net.py

--------------------------
在当前目录下执行python ./examples/ljftest_cifar10_DenseNet/net_look.py 	查看网络，打印的
在当前目录下执行python ./examples/ljftest_cifar10_DenseNet/show.py	可以查看中间训练图和概率图

============画出网络结构图：============
在/home/ljf/caffe-master目录下执行:
python /home/ljf/caffe-master/python/draw_net.py ./examples/ljftest_cifar10_DenseNet/train.prototxt ./examples/ljftest_cifar10_DenseNet/net.png



============绘制训练过程的loss和accuracy曲线============
博客地址:http://blog.csdn.net/u013078356/article/details/51154847
1.在工程目录下新建文件夹Log， 复制caffe-master/tools/extra/parse_log.sh  caffe-master/tools/extra/extract_seconds.py和 caffe-master/tools/extra/plot_training_log.py.example这三个文件进去

train_caffenet.sh里的内容改为：
GLOG_logtostderr=0 GLOG_log_dir=examples/ljftest_cifar10_DenseNet/Log/ ./build/tools/caffe train --solver=examples/ljftest_cifar10_DenseNet/solver.prototxt $@

2.去掉caffe.ljf-ubuntu.root.log后面的一串，在/home/ljf/caffe-master/examples/ljftest_cifar10_DenseNet/Log目录下执行 ./parse_log.sh caffe.ljf-ubuntu.root.log
会在当前文件夹下生成一个.train文件和一个.test文件

3.在/home/ljf/caffe-master/examples/ljftest_cifar10_DenseNet/Log目录下
执行
./plot_training_log.py.example 0  acc1.png caffe.ljf-ubuntu.root.log
./plot_training_log.py.example 1  acc2.png caffe.ljf-ubuntu.root.log
./plot_training_log.py.example 2  test_loss1.png caffe.ljf-ubuntu.root.log 
./plot_training_log.py.example 3  test_loss2.png caffe.ljf-ubuntu.root.log 
./plot_training_log.py.example 4  learning_rate1.png caffe.ljf-ubuntu.root.log 
./plot_training_log.py.example 6  train_loss1.png caffe.ljf-ubuntu.root.log 
生成图片

Notes:  
    1. Supporting multiple logs.  
    2. Log file name must end with the lower-cased ".log".  
Supported chart types:  
    0: Test accuracy  vs. Iters  
    1: Test accuracy  vs. Seconds  
    2: Test loss  vs. Iters  
    3: Test loss  vs. Seconds  
    4: Train learning rate  vs. Iters  
    5: Train learning rate  vs. Seconds  
    6: Train loss  vs. Iters  
    7: Train loss  vs. Seconds 
