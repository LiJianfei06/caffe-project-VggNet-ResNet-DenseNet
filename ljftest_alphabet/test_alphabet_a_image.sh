#!/usr/bin/env sh
set -e

./build/examples/cpp_classification/classification.bin \
/home/ljf/caffe-master/examples/ljftest_alphabet/deploy.prototxt \
/home/ljf/caffe-master/examples/ljftest_alphabet/caffe_ljftest_train_iter_50000.caffemodel \
/home/ljf/caffe-master/examples/ljftest_alphabet/imagenet_mean.binaryproto /home/ljf/caffe-master/examples/ljftest_alphabet/test/labels.txt \
/home/ljf/caffe-master/examples/ljftest_alphabet/test/image/875.jpg
