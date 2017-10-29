#!/usr/bin/env sh
set -e

/home/ljf/caffe-master/build/tools/caffe.bin test -model=/home/ljf/caffe-master/examples/ljftest_alphabet_VggNet/train_val.prototxt -weights=/home/ljf/caffe-master/examples/ljftest_alphabet_VggNet/caffe_ljftest_train_iter_110000.caffemodel
