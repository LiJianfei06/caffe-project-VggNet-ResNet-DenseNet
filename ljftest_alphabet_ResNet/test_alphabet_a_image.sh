#!/usr/bin/env sh
set -e

./build/examples/cpp_classification/classification.bin \
/home/ljf/caffe-master/examples/ljftest_alphabet_ResNet/deploy.prototxt \
/home/ljf/caffe-master/examples/ljftest_alphabet_ResNet/snapshot_ResNet_iter_150000.caffemodel \
/home/ljf/caffe-master/examples/ljftest_alphabet_ResNet/imagenet_mean.binaryproto /home/ljf/caffe-master/examples/ljftest_alphabet_ResNet/test/labels.txt \
/home/ljf/caffe-master/examples/ljftest_alphabet_ResNet/test/image/875.jpg
