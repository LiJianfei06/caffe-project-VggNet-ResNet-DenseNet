GLOG_logtostderr=0 GLOG_log_dir=examples/ljftest_cifar10_mobilenet/Log/ ./build/tools/caffe train --solver=examples/ljftest_cifar10_mobilenet/solver.prototxt 2>&1| tee examples/ljftest_cifar10_mobilenet/caffe.log$@

