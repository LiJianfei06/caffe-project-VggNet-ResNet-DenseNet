GLOG_logtostderr=0 GLOG_log_dir=examples/ljftest_cifar10_WRN/Log/ ./build/tools/caffe train --solver=examples/ljftest_cifar10_WRN/solver.prototxt 2>&1| tee examples/ljftest_cifar10_WRN/caffe.log$@

