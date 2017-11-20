#!/usr/bin/env sh
set -e

./build/examples/cpp_classification/classification.bin \
/home/ljf/caffe-master/examples/ljftest_cifar10_ResNet/deploy.prototxt \
/home/ljf/caffe-master/examples/ljftest_cifar10_ResNet/model_save/caffe_ljftest_train_iter_200000.caffemodel \
/home/ljf/caffe-master/examples/ljftest_cifar10_ResNet/imagenet_mean.binaryproto /home/ljf/caffe-master/examples/ljftest_cifar10_ResNet/test/labels.txt \
/home/ljf/caffe-master/examples/ljftest_cifar10_ResNet/test/image/horse5097.png
