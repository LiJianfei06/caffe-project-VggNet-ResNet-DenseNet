#!/usr/bin/env sh
set -e

/home/ljf/caffe-master/build/tools/caffe.bin test -model=/home/ljf/caffe-master/examples/ljftest_alphabet/train_val.prototxt -weights=/home/ljf/caffe-master/examples/ljftest_alphabet/caffe_ljftest_train_iter_50000.caffemodel
