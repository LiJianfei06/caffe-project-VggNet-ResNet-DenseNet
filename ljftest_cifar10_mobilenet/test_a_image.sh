#!/usr/bin/env sh
set -e

./build/examples/cpp_classification/classification.bin \
/home/ljf/caffe-master/examples/ljftest_cifar10_mobilenet/deploy.prototxt \
/home/ljf/caffe-master/examples/ljftest_cifar10_mobilenet/model_save/caffe_ljftest_train_iter_100000.caffemodel \
/home/ljf/caffe-master/examples/ljftest_cifar10_mobilenet/imagenet_mean.binaryproto /home/ljf/caffe-master/examples/ljftest_cifar10_mobilenet/test/labels.txt \
/home/ljf/caffe-master/examples/ljftest_cifar10_mobilenet/test/image/horse5097.png
