#!/usr/bin/env sh
set -e

./build/examples/cpp_classification/classification.bin \
/home/ljf/caffe-master/examples/ljftest_cifar10_WRN/deploy.prototxt \
/home/ljf/caffe-master/examples/ljftest_cifar10_WRN/model_save/caffe_ljftest_train_iter_100000.caffemodel \
/home/ljf/caffe-master/examples/ljftest_cifar10_WRN/imagenet_mean.binaryproto /home/ljf/caffe-master/examples/ljftest_cifar10_WRN/test/labels.txt \
/home/ljf/caffe-master/examples/ljftest_cifar10_WRN/test/image/horse5097.png
