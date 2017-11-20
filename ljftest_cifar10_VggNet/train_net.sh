GLOG_logtostderr=0 GLOG_log_dir=examples/ljftest_cifar10_VggNet_BN/Log/ ./build/tools/caffe train --solver=examples/ljftest_cifar10_VggNet_BN/solver.prototxt 2>&1| tee examples/ljftest_cifar10_VggNet_BN/caffe.log$@

