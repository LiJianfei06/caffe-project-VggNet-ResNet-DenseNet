GLOG_logtostderr=0 GLOG_log_dir=examples/ljftest_cifar10_ResNet_pool/Log/ ./build/tools/caffe train --solver=examples/ljftest_cifar10_ResNet_pool/solver.prototxt 2>&1| tee examples/ljftest_cifar10_ResNet_pool/caffe.log$@

