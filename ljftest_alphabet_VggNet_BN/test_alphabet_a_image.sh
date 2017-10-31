#!/usr/bin/env sh
set -e

./build/examples/cpp_classification/classification.bin \
/home/ljf/caffe-master/examples/ljftest_alphabet_VggNet_BN/deploy.prototxt \
/home/ljf/caffe-master/examples/ljftest_alphabet_VggNet_BN/caffe_ljftest_train_iter_192000.caffemodel \
/home/ljf/caffe-master/examples/ljftest_alphabet_VggNet_BN/imagenet_mean.binaryproto /home/ljf/caffe-master/examples/ljftest_alphabet_VggNet_BN/test/labels.txt \
/home/ljf/caffe-master/examples/ljftest_alphabet_VggNet_BN/test/image/875.jpg
