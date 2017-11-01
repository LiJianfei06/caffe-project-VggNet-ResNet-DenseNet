#!/usr/bin/env sh
set -e

/home/ljf/caffe-master/build/tools/caffe.bin test -model=/home/ljf/caffe-master/examples/ljftest_alphabet_ResNet/test.prototxt -weights=/home/ljf/caffe-master/examples/ljftest_alphabet_ResNet/snapshot_ResNet_iter_150000.caffemodel
