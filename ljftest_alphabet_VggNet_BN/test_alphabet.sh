#!/usr/bin/env sh
set -e

/home/ljf/caffe-master/build/tools/caffe.bin test -model=/home/ljf/caffe-master/examples/ljftest_alphabet_VggNet_BN/test.prototxt -weights=/home/ljf/caffe-master/examples/ljftest_alphabet_VggNet_BN/caffe_ljftest_train_iter_192000.caffemodel
