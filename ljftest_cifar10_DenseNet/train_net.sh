GLOG_logtostderr=0 GLOG_log_dir=examples/ljftest_cifar10_DenseNet/Log/ ./build/tools/caffe train --solver=examples/ljftest_cifar10_DenseNet/solver.prototxt 2>&1| tee examples/ljftest_cifar10_DenseNet/caffe.log$@

