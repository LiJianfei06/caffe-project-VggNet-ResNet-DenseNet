#!/usr/bin/env sh
set -e

./build/examples/cpp_classification/classification.bin \
/home/ljf/caffe-master/examples/ljftest_alphabet_VggNet/deploy.prototxt \
/home/ljf/caffe-master/examples/ljftest_alphabet_VggNet/caffe_ljftest_train_iter_110000.caffemodel \
/home/ljf/caffe-master/examples/ljftest_alphabet_VggNet/imagenet_mean.binaryproto /home/ljf/caffe-master/examples/ljftest_alphabet_VggNet/test/labels.txt \
/home/ljf/caffe-master/examples/ljftest_alphabet_VggNet/test/image/875.jpg
