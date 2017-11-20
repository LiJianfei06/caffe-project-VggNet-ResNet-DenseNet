#!/usr/bin/env sh
set -e

/home/ljf/caffe-master/build/tools/caffe.bin test -gpu 0 -model=/home/ljf/caffe-master/examples/ljftest_cifar10_VggNet/test.prototxt -weights=/home/ljf/caffe-master/examples/ljftest_cifar10_VggNet/model_save/caffe_ljftest_train_iter_190000.caffemodel
