#!/usr/bin/env sh
set -e

./build/examples/cpp_classification/classification.bin \
/home/ljf/caffe-master/examples/ljftest_cifar10_DenseNet/deploy.prototxt \
/home/ljf/caffe-master/examples/ljftest_cifar10_DenseNet/model_save/caffe_ljftest_train_iter_100000.caffemodel \
/home/ljf/caffe-master/examples/ljftest_cifar10_DenseNet/imagenet_mean.binaryproto /home/ljf/caffe-master/examples/ljftest_cifar10_DenseNet/test/labels.txt \
/home/ljf/caffe-master/examples/ljftest_cifar10_DenseNet/test/image/horse5097.png
